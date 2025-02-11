"""
Parallelizes the IR for distributed execution and inserts MPI calls.
"""

import copy
import hashlib
import inspect
import operator
import sys
import warnings
from collections import defaultdict

import llvmlite.binding as ll
import numba
import numpy as np
from numba.core import ir, ir_utils, types
from numba.core.ir_utils import (
    build_definitions,
    compile_to_numba_ir,
    compute_cfg_from_blocks,
    dprint_func_ir,
    find_callname,
    find_const,
    find_topo_order,
    get_definition,
    guard,
    is_get_setitem,
    mk_alloc,
    mk_unique_var,
    remove_dead,
    rename_labels,
    replace_arg_nodes,
    require,
    simplify,
)
from numba.parfors.parfor import (
    Parfor,
    _lower_parfor_sequential_block,
    get_parfor_params,
    unwrap_parfor_blocks,
    wrap_parfor_blocks,
)

import bodo
import bodo.utils.utils
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.time_ext import TimeType
from bodo.io import csv_cpp
from bodo.ir.connector import log_limit_pushdown
from bodo.libs.distributed_api import Reduce_Type
from bodo.libs.str_ext import (
    string_type,
    unicode_to_utf8,
    unicode_to_utf8_and_len,
)
from bodo.transforms.distributed_analysis import (
    DistributedAnalysis,
    Distribution,
    _get_array_accesses,
    _is_transposed_array,
    get_reduce_op,
)
from bodo.transforms.table_column_del_pass import remove_dead_table_columns
from bodo.utils.transform import (
    compile_func_single_block,
    get_call_expr_arg,
    get_const_value_inner,
    set_call_expr_arg,
    set_last_arg_to_true,
)
from bodo.utils.typing import (
    BodoError,
    decode_if_dict_array,
    get_overload_const_bool,
    get_overload_const_str,
    is_overload_constant_bool,
    is_overload_constant_str,
    is_str_arr_type,
    list_cumulative,
    to_str_arr_if_dict_array,
)
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
    is_ml_support_loaded,
    is_np_array_typ,
    is_slice_equiv_arr,
    is_whole_slice,
)

ll.add_symbol("csv_output_is_dir", csv_cpp.csv_output_is_dir)


distributed_run_extensions = {}

# analysis data for debugging
dist_analysis = None
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
        self,
        func_ir: ir.FunctionIR,
        typingctx,
        targetctx,
        typemap,
        calltypes,
        return_type,
        metadata,
        flags,
    ):
        self.func_ir = func_ir
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.typemap = typemap
        self.calltypes = calltypes
        self.return_type = return_type
        # Loc object of current location being translated
        self.curr_loc = self.func_ir.loc
        self.metadata = metadata
        self.flags = flags
        self.arr_analysis = numba.parfors.array_analysis.ArrayAnalysis(
            self.typingctx, self.func_ir, self.typemap, self.calltypes
        )

        self._dist_analysis = None
        # For each 1D parfor, map index variable name for the first dimension loop to
        # distributed start variable of the parfor
        self._1D_parfor_starts = {}
        # same as above but for 1D_Var parfors
        self._1D_Var_parfor_starts = {}
        # map 1D_Var arrays to index variable names for 1D_Var array accesses
        self._1D_Var_array_accesses = defaultdict(list)
        # keep start vars for 1D dist to reuse in parfor loop array accesses
        self._start_vars = {}
        # keep local versions of reduce variables to enable converting variable size
        # string allocations to local chunk size
        self._local_reduce_vars = {}

    def run(self):
        """Run distributed pass transforms"""
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
            self.return_type,
            self.typingctx,
            self.metadata,
            self.flags,
            self.arr_analysis,
        )
        self._dist_analysis = dist_analysis_pass.run()

        self._parallel_accesses = dist_analysis_pass._parallel_accesses
        if debug_prints():  # pragma: no cover
            print("distributions: ", self._dist_analysis)

        self.func_ir._definitions = build_definitions(self.func_ir.blocks)

        typing_info = bodo.compiler.TypingInfo(
            self.typingctx,
            self.targetctx,
            self.typemap,
            self.calltypes,
            self.func_ir.loc,
        )

        # Run column pruning and dead column elimination until there are no changes.
        # This updates table source nodes. Iterative convergence between Dead Col Elimination and Dead Code Elimination
        # is needed to enable pushing column pruning into IO nodes in some situations.
        # See: https://bodo.atlassian.net/wiki/spaces/B/pages/1064927240/Removing+dead+setitem+s+and+iterative+convergence+between+Dead+code+elimination+and+dead+column+elimination+WIP
        flag = True
        while flag:
            deadcode_elim_can_make_changes = remove_dead_table_columns(
                self.func_ir,
                self.typemap,
                typing_info,
                self._dist_analysis,
            )
            deadcode_eliminated = False
            # If dead columns are pruned, run dead code elimination until there are no changes
            if deadcode_elim_can_make_changes:
                # We extend numba's dead code elimination, through both remove_dead_extensions and in numba_compat.
                # So we can use their remove_dead
                while remove_dead(
                    self.func_ir.blocks,
                    self.func_ir.arg_names,
                    self.func_ir,
                    self.typemap,
                ):
                    deadcode_eliminated |= True
            # If we eliminated dead code, run these two passes again, and mark that we may need to
            # rerun typing pass to perform filter pushdown
            flag = deadcode_eliminated

        # transform
        self._gen_init_code(self.func_ir.blocks)
        self.func_ir.blocks = self._run_dist_pass(self.func_ir.blocks)

        while remove_dead(
            self.func_ir.blocks, self.func_ir.arg_names, self.func_ir, self.typemap
        ):
            pass
        dprint_func_ir(self.func_ir, "after distributed pass")
        lower_parfor_sequential(
            self.typingctx, self.func_ir, self.typemap, self.calltypes, self.metadata
        )
        if bodo.multithread_mode:
            # parfor params need to be updated for multithread_mode since some
            # new variables like alloc_start are introduced by distributed pass
            # and are used in later parfors
            _parfor_ids = get_parfor_params(
                self.func_ir.blocks, True, defaultdict(list)
            )

        # save data for debug and test
        global dist_analysis
        dist_analysis = self._dist_analysis
        return self._dist_analysis.ret_type

    def _run_dist_pass(self, blocks, init_avail=None):
        # init liveness info
        cfg = compute_cfg_from_blocks(blocks)
        all_avail_vars = find_available_vars(blocks, cfg, init_avail)
        topo_order = find_topo_order(blocks)
        for label in topo_order:
            block = blocks[label]
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
                    if isinstance(
                        inst,
                        (
                            bodo.ir.parquet_ext.ParquetReader,
                            bodo.ir.iceberg_ext.IcebergReader,
                        ),
                    ) or (
                        isinstance(inst, bodo.ir.sql_ext.SqlReader)
                        and inst.db_type == "snowflake"
                        # If streaming SQL has pushed down another limit its not necessarily
                        # safe to push down another limit as there may be a filter in the query.
                        and (inst.chunksize is None or inst.limit is None)
                    ):
                        # check if getting shape and/or head of parquet dataset is
                        # enough for the program
                        meta_head_only_info = guard(self._try_meta_head, inst)
                        # save meta for testing in test_io.py
                        self.func_ir.meta_head_only_info = meta_head_only_info
                        out_nodes = f(
                            inst,
                            self._dist_analysis.array_dists,
                            self.typemap,
                            self.calltypes,
                            self.typingctx,
                            self.targetctx,
                            is_independent=False,  # is_independent is False by default
                            meta_head_only_info=meta_head_only_info,
                        )
                    else:
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
                    self._fix_set_node_sig(inst)
                    out_nodes = []
                    index_var = get_getsetitem_index_var(inst, self.typemap, out_nodes)
                    out_nodes += self._run_getsetitem(
                        inst.target, index_var, inst, inst, equiv_set, avail_vars
                    )
                elif isinstance(inst, ir.SetAttr):
                    self._fix_set_node_sig(inst)
                elif isinstance(inst, ir.Return):
                    out_nodes = [inst]
                elif isinstance(inst, ir.Print):
                    out_nodes = self._run_print(inst, equiv_set)

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
            return self._run_array_shape(inst.target, rhs.value, equiv_set, avail_vars)

        # array.size
        if (
            rhs.op == "getattr"
            and rhs.attr == "size"
            and self._is_1D_or_1D_Var_arr(rhs.value.name)
        ):
            return self._run_array_size(inst.target, rhs.value, equiv_set, avail_vars)

        # array.T on 2D array
        if (
            rhs.op == "getattr"
            and rhs.attr == "T"
            and self._is_1D_or_1D_Var_arr(rhs.value.name)
            and isinstance(self.typemap[rhs.value.name], types.Array)
            and self.typemap[rhs.value.name].ndim == 2
        ):
            return self._run_array_transpose(inst, rhs.value)

        # index.nbytes,
        # bodo_arrays.nbytes
        # (BooleanArrayType, DecimalArrayType, IntegerArrayType, ...)
        if (
            rhs.op == "getattr"
            and rhs.attr == "nbytes"
            and self._is_1D_or_1D_Var_arr(rhs.value.name)
        ):
            return [inst] + compile_func_single_block(
                eval("lambda r: bodo.libs.distributed_api.dist_reduce(r.nbytes, _op)"),
                (rhs.value,),
                inst.target,
                self,
                extra_globals={"_op": np.int32(Reduce_Type.Sum.value)},
            )
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
                eval("lambda r: bodo.libs.distributed_api.dist_reduce(r._stop, _op)"),
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
                eval("lambda r: bodo.libs.distributed_api.dist_reduce(r._start, _op)"),
                (rhs.value,),
                inst.target,
                self,
                extra_globals={"_op": np.int32(Reduce_Type.Min.value)},
            )

        return [inst]

    def _run_call(self, assign: ir.Assign, equiv_set, avail_vars):
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

        if fdef == ("scalar_optional_getitem", "bodo.utils.indexing"):
            return self._run_getitem_scalar_optional(assign, equiv_set, avail_vars)

        if (
            fdef == ("table_filter", "bodo.hiframes.table")
            and self._is_1D_or_1D_Var_arr(lhs)
            and isinstance(self.typemap[rhs.args[1].name], types.SliceType)
        ):
            # Code generated for the table filter function.
            # If we have a distributed slice then we need to
            # create a local slice.
            in_table = rhs.args[0]
            index_var = rhs.args[1]
            # TODO: Handle the full set of slice options
            start_var, nodes = self._get_dist_start_var(in_table, equiv_set, avail_vars)
            size_var = self._get_dist_var_len(in_table, nodes, equiv_set, avail_vars)
            new_nodes = compile_func_single_block(
                eval(
                    "lambda slice_index, start, tot_len: bodo.libs.distributed_api.get_local_slice("
                    "    slice_index,"
                    "    start,"
                    "    tot_len,"
                    ")"
                ),
                [index_var, start_var, size_var],
                ret_var=None,
                typing_info=self,
            )
            new_call = ir.Expr.call(
                rhs.func, [in_table, new_nodes[-1].target], rhs.kws, rhs.loc
            )
            new_assign = ir.Assign(new_call, assign.target, loc)
            # Update the call types
            self.calltypes[new_call] = self.calltypes[rhs]
            del self.calltypes[rhs]
            return nodes + new_nodes + [new_assign]

        if fdef == ("interp_bin_search", "bodo.libs.array_kernels"):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                set_last_arg_to_true(self, assign.value)

        if fdef == ("sum_decimal_array", "bodo.libs.decimal_arr_ext"):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                set_last_arg_to_true(self, assign.value)

        if fdef == ("bodosql_listagg", "bodosql.kernels.listagg"):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                set_last_arg_to_true(self, assign.value)

        if fdef == (
            "generate_table_nbytes",
            "bodo.utils.table_utils",
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if (
            fdef == ("init_join_state", "bodo.libs.streaming.join")
            and lhs in self._dist_analysis.array_dists
        ):
            build_dist, probe_dist = self._dist_analysis.array_dists[lhs]
            distributed_dists = (Distribution.OneD, Distribution.OneD_Var)
            # Check if the build and probe arrays are distributed
            if build_dist in distributed_dists or probe_dist in distributed_dists:
                build_parallel = build_dist in distributed_dists
                probe_parallel = probe_dist in distributed_dists
                # Whichever is distributed update the args in the signature to True.
                call_type = self.calltypes.pop(rhs)
                assert call_type.args[-2] == types.Omitted(False) and call_type.args[
                    -1
                ] == types.Omitted(False)
                new_sig = self.typemap[rhs.func.name].get_call_type(
                    self.typingctx,
                    call_type.args[:-2]
                    + (types.Omitted(build_parallel), types.Omitted(probe_parallel)),
                    {},
                )
                new_sig = new_sig.replace(return_type=call_type.return_type)
                self.calltypes[rhs] = new_sig
                return [assign]

        if fdef in (
            (
                "snowflake_writer_init",
                "bodo.io.snowflake_write",
            ),
            (
                "iceberg_writer_init",
                "bodo.io.iceberg.stream_iceberg_write",
            ),
            (
                "parquet_writer_init",
                "bodo.io.stream_parquet_write",
            ),
            (
                "init_groupby_state",
                "bodo.libs.streaming.groupby",
            ),
            (
                "init_grouping_sets_state",
                "bodo.libs.streaming.groupby",
            ),
            (
                "init_stream_sort_state",
                "bodo.libs.streaming.sort",
            ),
            (
                "init_union_state",
                "bodo.libs.streaming.union",
            ),
            (
                "init_window_state",
                "bodo.libs.streaming.window",
            ),
        ) and self._is_1D_or_1D_Var_arr(lhs):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if (
            func_name == "fit"
            and "bodo.ml_support.xgb_ext" in sys.modules
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                (
                    bodo.ml_support.xgb_ext.BodoXGBClassifierType,
                    bodo.ml_support.xgb_ext.BodoXGBRegressorType,
                ),
            )
        ):  # pragma: no cover
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                set_last_arg_to_true(self, assign.value)
                return [assign]

        if (
            func_name == "fit"
            and is_ml_support_loaded()
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                (
                    bodo.ml_support.sklearn_cluster_ext.BodoKMeansClusteringType,
                    bodo.ml_support.sklearn_ensemble_ext.BodoRandomForestClassifierType,
                    bodo.ml_support.sklearn_ensemble_ext.BodoRandomForestRegressorType,
                    bodo.ml_support.sklearn_linear_model_ext.BodoSGDClassifierType,
                    bodo.ml_support.sklearn_linear_model_ext.BodoSGDRegressorType,
                    bodo.ml_support.sklearn_linear_model_ext.BodoLogisticRegressionType,
                    bodo.ml_support.sklearn_linear_model_ext.BodoLinearRegressionType,
                    bodo.ml_support.sklearn_linear_model_ext.BodoLassoType,
                    bodo.ml_support.sklearn_linear_model_ext.BodoRidgeType,
                    bodo.ml_support.sklearn_naive_bayes_ext.BodoMultinomialNBType,
                    bodo.ml_support.sklearn_svm_ext.BodoLinearSVCType,
                    bodo.ml_support.sklearn_preprocessing_ext.BodoPreprocessingOneHotEncoderType,
                    bodo.ml_support.sklearn_preprocessing_ext.BodoPreprocessingStandardScalerType,
                    bodo.ml_support.sklearn_preprocessing_ext.BodoPreprocessingMaxAbsScalerType,
                    bodo.ml_support.sklearn_preprocessing_ext.BodoPreprocessingMinMaxScalerType,
                    bodo.ml_support.sklearn_preprocessing_ext.BodoPreprocessingRobustScalerType,
                    bodo.ml_support.sklearn_preprocessing_ext.BodoPreprocessingLabelEncoderType,
                ),
            )
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if (
            func_name == "partial_fit"
            and "bodo.ml_support.sklearn_preprocessing_ext" in sys.modules
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.ml_support.sklearn_preprocessing_ext.BodoPreprocessingMaxAbsScalerType,
            )
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if (
            func_name == "score"
            and is_ml_support_loaded()
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                (
                    bodo.ml_support.sklearn_cluster_ext.BodoKMeansClusteringType,
                    bodo.ml_support.sklearn_ensemble_ext.BodoRandomForestClassifierType,
                    bodo.ml_support.sklearn_ensemble_ext.BodoRandomForestRegressorType,
                    bodo.ml_support.sklearn_linear_model_ext.BodoSGDClassifierType,
                    bodo.ml_support.sklearn_linear_model_ext.BodoSGDRegressorType,
                    bodo.ml_support.sklearn_linear_model_ext.BodoLogisticRegressionType,
                    bodo.ml_support.sklearn_linear_model_ext.BodoLinearRegressionType,
                    bodo.ml_support.sklearn_linear_model_ext.BodoLassoType,
                    bodo.ml_support.sklearn_linear_model_ext.BodoRidgeType,
                    bodo.ml_support.sklearn_naive_bayes_ext.BodoMultinomialNBType,
                    bodo.ml_support.sklearn_svm_ext.BodoLinearSVCType,
                ),
            )
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                set_last_arg_to_true(self, assign.value)
                return [assign]
        if (
            func_name == "fit_transform"
            and (
                "bodo.ml_support.sklearn_feature_extraction_ext" in sys.modules
                or "bodo.ml_support.sklearn_preprocessing_ext" in sys.modules
            )
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                (
                    bodo.ml_support.sklearn_preprocessing_ext.BodoPreprocessingLabelEncoderType,
                    bodo.ml_support.sklearn_feature_extraction_ext.BodoFExtractHashingVectorizerType,
                    bodo.ml_support.sklearn_feature_extraction_ext.BodoFExtractCountVectorizerType,
                ),
            )
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                set_last_arg_to_true(self, assign.value)
                return [assign]

        if (
            func_mod == "sklearn.utils"
            and func_name == "shuffle"
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            import sklearn

            rhs = assign.value
            kws = dict(rhs.kws)
            nodes = []

            data = get_call_expr_arg("sklearn.utils.shuffle", rhs.args, kws, 0, "data")

            # random_state argument
            random_state_var = ir.Var(
                assign.target.scope,
                mk_unique_var("shuffle_random_state"),
                rhs.loc,
            )
            nodes.append(ir.Assign(ir.Const(None, rhs.loc), random_state_var, rhs.loc))
            self.typemap[random_state_var.name] = types.none
            # random_state cannot be specified positionally
            random_state = get_call_expr_arg(
                "shuffle", rhs.args, kws, 1e6, "random_state", random_state_var
            )

            # n_samples argument
            n_samples_var = ir.Var(
                assign.target.scope,
                mk_unique_var("shuffle_n_samples"),
                rhs.loc,
            )
            nodes.append(ir.Assign(ir.Const(None, rhs.loc), n_samples_var, rhs.loc))
            self.typemap[n_samples_var.name] = types.none
            # n_samples cannot be specified positionally
            n_samples = get_call_expr_arg(
                "shuffle", rhs.args, kws, 1e6, "n_samples", n_samples_var
            )

            f = eval(
                "lambda data, random_state, n_samples: sklearn.utils.shuffle("
                "    data, random_state=random_state, n_samples=n_samples, _is_data_distributed=True"
                ")"
            )
            return nodes + compile_func_single_block(
                f,
                [data, random_state, n_samples],
                assign.target,
                self,
                extra_globals={"sklearn": sklearn},
            )

        if (
            func_mod in ("sklearn.metrics._classification", "sklearn.metrics")
            and func_name == "precision_score"
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                import sklearn

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

                f = eval(
                    "lambda y_true, y_pred, average: sklearn.metrics.precision_score("
                    "    y_true, y_pred, average=average, _is_data_distributed=True"
                    ")"
                )
                return nodes + compile_func_single_block(
                    f,
                    [y_true, y_pred, average],
                    assign.target,
                    self,
                    extra_globals={"sklearn": sklearn},
                )

        if (
            func_mod in ("sklearn.metrics._classification", "sklearn.metrics")
            and func_name == "recall_score"
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                import sklearn

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

                f = eval(
                    "lambda y_true, y_pred, average: sklearn.metrics.recall_score("
                    "    y_true, y_pred, average=average, _is_data_distributed=True"
                    ")"
                )
                return nodes + compile_func_single_block(
                    f,
                    [y_true, y_pred, average],
                    assign.target,
                    self,
                    extra_globals={"sklearn": sklearn},
                )

        if (
            func_mod in ("sklearn.metrics._classification", "sklearn.metrics")
            and func_name == "f1_score"
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                import sklearn

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

                f = eval(
                    "lambda y_true, y_pred, average: sklearn.metrics.f1_score("
                    "    y_true, y_pred, average=average, _is_data_distributed=True"
                    ")"
                )
                return nodes + compile_func_single_block(
                    f,
                    [y_true, y_pred, average],
                    assign.target,
                    self,
                    extra_globals={"sklearn": sklearn},
                )

        if (
            func_mod in ("sklearn.metrics._classification", "sklearn.metrics")
            and func_name == "log_loss"
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            import sklearn

            rhs = assign.value
            kws = dict(rhs.kws)
            nodes = []

            y_true = get_call_expr_arg(
                "sklearn.metrics.log_loss", rhs.args, kws, 0, "y_true"
            )
            y_pred = get_call_expr_arg(
                "sklearn.metrics.log_loss", rhs.args, kws, 1, "y_pred"
            )

            # eps argument
            eps_var = ir.Var(
                assign.target.scope,
                mk_unique_var("log_loss_eps"),
                rhs.loc,
            )
            nodes.append(ir.Assign(ir.Const(1e-15, rhs.loc), eps_var, rhs.loc))
            self.typemap[eps_var.name] = types.float64
            # eps cannot be specified positionally
            eps = get_call_expr_arg("log_loss", rhs.args, kws, 1e6, "eps", eps_var)

            # normalize argument
            normalize_var = ir.Var(
                assign.target.scope,
                mk_unique_var("log_loss_normalize"),
                rhs.loc,
            )
            nodes.append(ir.Assign(ir.Const(True, rhs.loc), normalize_var, rhs.loc))
            self.typemap[normalize_var.name] = types.BooleanLiteral(True)
            # normalize cannot be specified positionally
            normalize = get_call_expr_arg(
                "log_loss", rhs.args, kws, 1e6, "normalize", normalize_var
            )

            # sample_weight argument
            sample_weight_var = ir.Var(
                assign.target.scope,
                mk_unique_var("log_loss_sample_weight"),
                rhs.loc,
            )
            nodes.append(ir.Assign(ir.Const(None, rhs.loc), sample_weight_var, rhs.loc))
            self.typemap[sample_weight_var.name] = types.none
            # sample_weight cannot be specified positionally
            sample_weight = get_call_expr_arg(
                "log_loss", rhs.args, kws, 1e6, "sample_weight", sample_weight_var
            )

            # labels argument
            labels_var = ir.Var(
                assign.target.scope,
                mk_unique_var("log_loss_labels"),
                rhs.loc,
            )
            nodes.append(ir.Assign(ir.Const(None, rhs.loc), labels_var, rhs.loc))
            self.typemap[labels_var.name] = types.none
            # labels cannot be specified positionally
            labels = get_call_expr_arg(
                "log_loss", rhs.args, kws, 1e6, "labels", labels_var
            )

            f = eval(
                "lambda y_true, y_pred, eps, normalize, sample_weight, labels: sklearn.metrics.log_loss("
                "    y_true,"
                "    y_pred,"
                "    eps=eps,"
                "    normalize=normalize,"
                "    sample_weight=sample_weight,"
                "    labels=labels,"
                "    _is_data_distributed=True,"
                ")"
            )
            return nodes + compile_func_single_block(
                f,
                [y_true, y_pred, eps, normalize, sample_weight, labels],
                assign.target,
                self,
                extra_globals={"sklearn": sklearn},
            )

        if (
            func_mod in ("sklearn.metrics._classification", "sklearn.metrics")
            and func_name == "accuracy_score"
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            import sklearn

            rhs = assign.value
            kws = dict(rhs.kws)
            nodes = []

            y_true = get_call_expr_arg(
                "sklearn.metrics.accuracy_score", rhs.args, kws, 0, "y_true"
            )
            y_pred = get_call_expr_arg(
                "sklearn.metrics.accuracy_score", rhs.args, kws, 1, "y_pred"
            )

            ## normalize argument
            normalize_var = ir.Var(
                assign.target.scope,
                mk_unique_var("accuracy_score_normalize"),
                rhs.loc,
            )
            nodes.append(ir.Assign(ir.Const(True, rhs.loc), normalize_var, rhs.loc))
            self.typemap[normalize_var.name] = types.BooleanLiteral(True)
            # normalize cannot be specified positionally
            normalize = get_call_expr_arg(
                "accuracy_score", rhs.args, kws, 1e6, "normalize", normalize_var
            )

            # sample_weight argument
            sample_weight_var = ir.Var(
                assign.target.scope,
                mk_unique_var("accuracy_score_sample_weight"),
                rhs.loc,
            )
            nodes.append(ir.Assign(ir.Const(None, rhs.loc), sample_weight_var, rhs.loc))
            self.typemap[sample_weight_var.name] = types.none
            # sample_weight cannot be specified positionally
            sample_weight = get_call_expr_arg(
                "accuracy_score",
                rhs.args,
                kws,
                1e6,
                "sample_weight",
                sample_weight_var,
            )

            f = eval(
                "lambda y_true, y_pred, normalize, sample_weight: sklearn.metrics.accuracy_score("
                "    y_true,"
                "    y_pred,"
                "    normalize=normalize,"
                "    sample_weight=sample_weight,"
                "    _is_data_distributed=True,"
                ")"
            )
            return nodes + compile_func_single_block(
                f,
                [y_true, y_pred, normalize, sample_weight],
                assign.target,
                self,
                extra_globals={"sklearn": sklearn},
            )

        if (
            func_mod in ("sklearn.metrics._classification", "sklearn.metrics")
            and func_name == "confusion_matrix"
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                import sklearn

                rhs = assign.value
                kws = dict(rhs.kws)
                nodes = []

                # TODO Add error checking for argument data types

                y_true = get_call_expr_arg(
                    "sklearn.metrics.confusion_matrix", rhs.args, kws, 0, "y_true"
                )
                y_pred = get_call_expr_arg(
                    "sklearn.metrics.confusion_matrix", rhs.args, kws, 1, "y_pred"
                )

                # labels argument
                labels_var = ir.Var(
                    assign.target.scope,
                    mk_unique_var("confusion_matrix_labels"),
                    rhs.loc,
                )
                nodes.append(ir.Assign(ir.Const(None, rhs.loc), labels_var, rhs.loc))
                self.typemap[labels_var.name] = types.none
                # labels cannot be specified positionally
                labels = get_call_expr_arg(
                    "confusion_matrix",
                    rhs.args,
                    kws,
                    1e6,
                    "labels",
                    labels_var,
                )

                # sample_weight argument
                sample_weight_var = ir.Var(
                    assign.target.scope,
                    mk_unique_var("confusion_matrix_sample_weight"),
                    rhs.loc,
                )
                nodes.append(
                    ir.Assign(ir.Const(None, rhs.loc), sample_weight_var, rhs.loc)
                )
                self.typemap[sample_weight_var.name] = types.none
                # sample_weight cannot be specified positionally
                sample_weight = get_call_expr_arg(
                    "confusion_matrix",
                    rhs.args,
                    kws,
                    1e6,
                    "sample_weight",
                    sample_weight_var,
                )

                ## normalize argument
                normalize_var = ir.Var(
                    assign.target.scope,
                    mk_unique_var("confusion_matrix_normalize"),
                    rhs.loc,
                )
                nodes.append(ir.Assign(ir.Const(None, rhs.loc), normalize_var, rhs.loc))
                self.typemap[normalize_var.name] = types.none
                # normalize cannot be specified positionally
                normalize = get_call_expr_arg(
                    "confusion_matrix", rhs.args, kws, 1e6, "normalize", normalize_var
                )

                f = eval(
                    "lambda y_true, y_pred, labels, sample_weight, normalize: sklearn.metrics.confusion_matrix("
                    "    y_true,"
                    "    y_pred,"
                    "    labels=labels,"
                    "    sample_weight=sample_weight,"
                    "    normalize=normalize,"
                    "    _is_data_distributed=True,"
                    ")"
                )
                return nodes + compile_func_single_block(
                    f,
                    [y_true, y_pred, labels, sample_weight, normalize],
                    assign.target,
                    self,
                    extra_globals={"sklearn": sklearn},
                )

        if (
            func_mod in ("sklearn.metrics._regression", "sklearn.metrics")
            and func_name == "mean_squared_error"
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                import sklearn

                rhs = assign.value
                kws = dict(rhs.kws)
                nodes = []

                y_true = get_call_expr_arg(
                    "sklearn.metrics.mean_squared_error", rhs.args, kws, 0, "y_true"
                )
                y_pred = get_call_expr_arg(
                    "sklearn.metrics.mean_squared_error", rhs.args, kws, 1, "y_pred"
                )

                # sample_weight argument; since it cannot be specified positionally
                sample_weight_var = ir.Var(
                    assign.target.scope,
                    mk_unique_var("mean_squared_error_sample_weight"),
                    rhs.loc,
                )
                nodes.append(
                    ir.Assign(ir.Const(None, rhs.loc), sample_weight_var, rhs.loc)
                )
                self.typemap[sample_weight_var.name] = types.none
                sample_weight = get_call_expr_arg(
                    "mean_squared_error",
                    rhs.args,
                    kws,
                    1e6,
                    "sample_weight",
                    sample_weight_var,
                )

                # multioutput argument; since it cannot be specified positionally
                multioutput_var = ir.Var(
                    assign.target.scope,
                    mk_unique_var("mean_squared_error_multioutput"),
                    rhs.loc,
                )
                nodes.append(
                    ir.Assign(
                        ir.Const("uniform_average", rhs.loc), multioutput_var, rhs.loc
                    )
                )
                self.typemap[multioutput_var.name] = types.StringLiteral(
                    "uniform_average"
                )
                multioutput = get_call_expr_arg(
                    "mean_squared_error",
                    rhs.args,
                    kws,
                    1e6,
                    "multioutput",
                    multioutput_var,
                )

                # squared argument; since it cannot be specified positionally
                squared_var = ir.Var(
                    assign.target.scope,
                    mk_unique_var("mean_squared_error_squared"),
                    rhs.loc,
                )
                nodes.append(ir.Assign(ir.Const(True, rhs.loc), squared_var, rhs.loc))
                self.typemap[squared_var.name] = types.BooleanLiteral(True)
                squared = get_call_expr_arg(
                    "mean_squared_error", rhs.args, kws, 1e6, "squared", squared_var
                )

                f = eval(
                    "lambda y_true, y_pred, sample_weight, multioutput, squared: sklearn.metrics.mean_squared_error("
                    "    y_true,"
                    "    y_pred,"
                    "    sample_weight=sample_weight,"
                    "    multioutput=multioutput,"
                    "    squared=squared,"
                    "    _is_data_distributed=True,"
                    ")"
                )
                return nodes + compile_func_single_block(
                    f,
                    [y_true, y_pred, sample_weight, multioutput, squared],
                    assign.target,
                    self,
                    extra_globals={"sklearn": sklearn},
                )

        if (
            func_mod in ("sklearn.metrics._regression", "sklearn.metrics")
            and func_name == "mean_absolute_error"
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                import sklearn

                rhs = assign.value
                kws = dict(rhs.kws)
                nodes = []

                y_true = get_call_expr_arg(
                    "sklearn.metrics.mean_absolute_error", rhs.args, kws, 0, "y_true"
                )
                y_pred = get_call_expr_arg(
                    "sklearn.metrics.mean_absolute_error", rhs.args, kws, 1, "y_pred"
                )

                # sample_weight argument; since it cannot be specified positionally
                sample_weight_var = ir.Var(
                    assign.target.scope,
                    mk_unique_var("mean_absolute_error_sample_weight"),
                    rhs.loc,
                )
                nodes.append(
                    ir.Assign(ir.Const(None, rhs.loc), sample_weight_var, rhs.loc)
                )
                self.typemap[sample_weight_var.name] = types.none
                sample_weight = get_call_expr_arg(
                    "mean_absolute_error",
                    rhs.args,
                    kws,
                    1e6,
                    "sample_weight",
                    sample_weight_var,
                )

                # multioutput argument; since it cannot be specified positionally
                multioutput_var = ir.Var(
                    assign.target.scope,
                    mk_unique_var("mean_absolute_error_multioutput"),
                    rhs.loc,
                )
                nodes.append(
                    ir.Assign(
                        ir.Const("uniform_average", rhs.loc), multioutput_var, rhs.loc
                    )
                )
                self.typemap[multioutput_var.name] = types.StringLiteral(
                    "uniform_average"
                )
                multioutput = get_call_expr_arg(
                    "mean_absolute_error",
                    rhs.args,
                    kws,
                    1e6,
                    "multioutput",
                    multioutput_var,
                )

                f = eval(
                    "lambda y_true, y_pred, sample_weight, multioutput: sklearn.metrics.mean_absolute_error("
                    "    y_true,"
                    "    y_pred,"
                    "    sample_weight=sample_weight,"
                    "    multioutput=multioutput,"
                    "    _is_data_distributed=True,"
                    ")"
                )
                return nodes + compile_func_single_block(
                    f,
                    [y_true, y_pred, sample_weight, multioutput],
                    assign.target,
                    self,
                    extra_globals={"sklearn": sklearn},
                )

        if func_mod == "sklearn.metrics.pairwise" and func_name == "cosine_similarity":
            # Set last argument to True if X is distributed
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                set_last_arg_to_true(self, assign.value)

            # Set second-to-last argument to True if Y exists and is distributed
            # Y could be passed in as the second positional arg, or a kwarg if not None.
            # We use `fold_argument_types` to handle both cases
            pysig = self.calltypes[rhs].pysig
            folded_args = bodo.utils.transform.fold_argument_types(
                pysig, rhs.args, rhs.kws
            )
            y_arg = folded_args[1]
            if isinstance(y_arg, ir.Var) and self._is_1D_or_1D_Var_arr(y_arg.name):
                self._set_second_last_arg_to_true(assign.value)

            return [assign]

        if (
            func_name == "split"
            and "bodo.ml_support.sklearn_model_selection_ext" in sys.modules
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.ml_support.sklearn_model_selection_ext.BodoModelSelectionKFoldType,
            )
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            # Not checking get_n_splits for KFold since it might not have a first arg
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if (
            func_name in ("split", "get_n_splits")
            and "bodo.ml_support.sklearn_model_selection_ext" in sys.modules
            and isinstance(func_mod, numba.core.ir.Var)
            and isinstance(
                self.typemap[func_mod.name],
                bodo.ml_support.sklearn_model_selection_ext.BodoModelSelectionLeavePOutType,
            )
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if (
            func_mod in ("sklearn.model_selection._split", "sklearn.model_selection")
            and func_name == "train_test_split"
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                import sklearn

                rhs = assign.value
                kws = dict(rhs.kws)
                nodes = []
                data = get_call_expr_arg(
                    "sklearn.model_selection.train_test_split", rhs.args, kws, 0, "data"
                )
                labels = get_call_expr_arg(
                    "sklearn.model_selection.train_test_split",
                    rhs.args,
                    kws,
                    1,
                    "labels",
                )
                test_size_var = ir.Var(
                    assign.target.scope, mk_unique_var("test_size_var"), rhs.loc
                )
                nodes.append(ir.Assign(ir.Const(0.25, rhs.loc), test_size_var, rhs.loc))
                self.typemap[test_size_var.name] = types.none
                test_size = get_call_expr_arg(
                    "train_test_split", rhs.args, kws, 1e6, "test_size", test_size_var
                )

                train_size_var = ir.Var(
                    assign.target.scope, mk_unique_var("train_size_var"), rhs.loc
                )
                nodes.append(
                    ir.Assign(ir.Const(0.75, rhs.loc), train_size_var, rhs.loc)
                )
                self.typemap[train_size_var.name] = types.none
                train_size = get_call_expr_arg(
                    "train_test_split", rhs.args, kws, 1e6, "train_size", train_size_var
                )

                shuffle_var = ir.Var(
                    assign.target.scope, mk_unique_var("shuffle_var"), rhs.loc
                )
                nodes.append(ir.Assign(ir.Const(True, rhs.loc), shuffle_var, rhs.loc))
                self.typemap[shuffle_var.name] = types.BooleanLiteral(True)
                shuffle = get_call_expr_arg(
                    "train_test_split", rhs.args, kws, 1e6, "shuffle", shuffle_var
                )

                random_state_var = ir.Var(
                    assign.target.scope, mk_unique_var("random_state_var"), rhs.loc
                )
                # Q: int or randomstate instance?
                nodes.append(ir.Assign(ir.Const(0, rhs.loc), random_state_var, rhs.loc))
                self.typemap[random_state_var.name] = types.IntegerLiteral(0)
                random_state = get_call_expr_arg(
                    "train_test_split",
                    rhs.args,
                    kws,
                    1e6,
                    "random_state",
                    random_state_var,
                )
                f = eval(
                    "lambda data, labels, test_size, train_size, shuffle, random_state: sklearn.model_selection.train_test_split("
                    "    data,"
                    "    labels,"
                    "    test_size=test_size,"
                    "    train_size=train_size,"
                    "    shuffle=shuffle,"
                    "    random_state=random_state,"
                    "    _is_data_distributed=True,"
                    ")"
                )
                return nodes + compile_func_single_block(
                    f,
                    [data, labels, test_size, train_size, shuffle, random_state],
                    assign.target,
                    self,
                    extra_globals={"sklearn": sklearn},
                )

        if (
            func_mod in ("sklearn.metrics._regression", "sklearn.metrics")
            and func_name == "r2_score"
        ):
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                import sklearn

                rhs = assign.value
                kws = dict(rhs.kws)
                nodes = []

                y_true = get_call_expr_arg(
                    "sklearn.metrics.r2_score", rhs.args, kws, 0, "y_true"
                )
                y_pred = get_call_expr_arg(
                    "sklearn.metrics.r2_score", rhs.args, kws, 1, "y_pred"
                )

                # sample_weight argument; since it cannot be specified positionally
                sample_weight_var = ir.Var(
                    assign.target.scope,
                    mk_unique_var("r2_score_sample_weight"),
                    rhs.loc,
                )
                nodes.append(
                    ir.Assign(ir.Const(None, rhs.loc), sample_weight_var, rhs.loc)
                )
                self.typemap[sample_weight_var.name] = types.none
                sample_weight = get_call_expr_arg(
                    "r2_score",
                    rhs.args,
                    kws,
                    1e6,
                    "sample_weight",
                    sample_weight_var,
                )

                # multioutput argument; since it cannot be specified positionally
                multioutput_var = ir.Var(
                    assign.target.scope,
                    mk_unique_var("r2_score_multioutput"),
                    rhs.loc,
                )
                nodes.append(
                    ir.Assign(
                        ir.Const("uniform_average", rhs.loc), multioutput_var, rhs.loc
                    )
                )
                self.typemap[multioutput_var.name] = types.StringLiteral(
                    "uniform_average"
                )
                multioutput = get_call_expr_arg(
                    "r2_score",
                    rhs.args,
                    kws,
                    1e6,
                    "multioutput",
                    multioutput_var,
                )

                f = eval(
                    "lambda y_true, y_pred, sample_weight, multioutput: sklearn.metrics.r2_score("
                    "    y_true,"
                    "    y_pred,"
                    "    sample_weight=sample_weight,"
                    "    multioutput=multioutput,"
                    "    _is_data_distributed=True,"
                    ")"
                )
                return nodes + compile_func_single_block(
                    f,
                    [y_true, y_pred, sample_weight, multioutput],
                    assign.target,
                    self,
                    extra_globals={"sklearn": sklearn},
                )

        # divide 1D alloc
        # XXX allocs should be matched before going to _run_call_np
        if self._is_1D_arr(lhs) and is_alloc_callname(func_name, func_mod):
            # XXX for pre_alloc_string_array(n, nc), we assume nc is local
            # value (updated only in parfor like _str_replace_regex_impl)
            # get local number of characters for string allocation if there is a
            # reduction
            if fdef in (
                ("pre_alloc_string_array", "bodo.libs.str_arr_ext"),
                ("pre_alloc_binary_array", "bodo.libs.binary_arr_ext"),
            ):
                n_char_var = rhs.args[1]
                if n_char_var.name in self._local_reduce_vars:
                    rhs.args[1] = self._local_reduce_vars[n_char_var.name]
                    if isinstance(self.typemap[n_char_var.name], types.Literal):
                        self._set_ith_arg_to_unliteral(rhs, 1)

            size_var = rhs.args[0]
            out, new_size_var = self._run_alloc(size_var, scope, loc)
            # empty_inferred is tuple for some reason
            rhs.args = list(rhs.args)
            rhs.args[0] = new_size_var
            if isinstance(self.typemap[size_var.name], types.Literal):
                self._set_ith_arg_to_unliteral(rhs, 0)
            out.append(assign)
            return out

        # fix 1D_Var allocs in case global len of another 1DVar is used
        if self._is_1D_Var_arr(lhs) and is_alloc_callname(func_name, func_mod):
            # get local number of characters for string allocation if there is a
            # reduction
            if fdef in (
                ("pre_alloc_string_array", "bodo.libs.str_arr_ext"),
                ("pre_alloc_binary_array", "bodo.libs.binary_arr_ext"),
            ):
                n_char_var = rhs.args[1]
                if n_char_var.name in self._local_reduce_vars:
                    rhs.args[1] = self._local_reduce_vars[n_char_var.name]
                    if isinstance(self.typemap[n_char_var.name], types.Literal):
                        self._set_ith_arg_to_unliteral(rhs, 1)

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
            if isinstance(self.typemap[size_var.name], types.Literal):
                self._set_ith_arg_to_unliteral(rhs, 0)
            out.append(assign)
            return out

        # Distributed handling of list_to_array requires handling similar to a distributed allocation.
        if fdef == (
            "list_to_array",
            "bodo.utils.conversion",
        ) and self._is_1D_arr(lhs):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        # array_to_repeated_array_item_array() is similar to an allocation and its size
        # argument needs handled similarly
        if fdef in (
            (
                "array_to_repeated_array_item_array",
                "bodo.libs.array_item_arr_ext",
            ),
            (
                "scalar_to_map_array",
                "bodo.libs.map_arr_ext",
            ),
            (
                "scalar_to_struct_array",
                "bodo.libs.struct_arr_ext",
            ),
        ):
            out = []
            size_var = rhs.args[1]
            if self._is_1D_arr(lhs):
                rhs.args[1] = self._get_1D_count(size_var, out)
            elif self._is_1D_Var_arr(lhs):
                rhs.args[1] = self._get_1D_Var_size(
                    size_var, equiv_set, avail_vars, out
                )

            out.append(assign)
            return out

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

        # BooleanArray.func calls
        if (
            isinstance(func_mod, ir.Var)
            and self.typemap[func_mod.name] == bodo.boolean_array_type
        ):
            return self._run_call_boolean_array(func_mod, func_name, assign)

        # df.func calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], DataFrameType
        ):
            return self._run_call_df(lhs, func_mod, func_name, assign, rhs.args)

        # series.func calls
        if isinstance(func_mod, ir.Var) and isinstance(
            self.typemap[func_mod.name], SeriesType
        ):
            return self._run_call_series(lhs, func_mod, func_name, assign, rhs.args)

        if fdef == ("permutation", "numpy.random"):
            if self.typemap[rhs.args[0].name] == types.int64:
                return self._run_permutation_int(assign, rhs.args)

        # len(A) if A is 1D or 1D_Var
        if (
            fdef == ("len", "builtins")
            and rhs.args
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
            # no need for transformation for len of distributed List/Dict
            and not isinstance(
                self.typemap[rhs.args[0].name], (types.List, types.DictType)
            )
        ):
            arr = rhs.args[0]
            # concat reduction variables don't need transformation
            # see test_concat_reduction
            if arr.name in self._dist_analysis.concat_reduce_varnames:
                return [assign]
            nodes = []
            assign.value = self._get_dist_var_len(arr, nodes, equiv_set, avail_vars)
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
            func_mod == "bodo.io.h5_api"
            and func_name in ("h5read", "h5write", "h5read_filter")
            and self._is_1D_arr(rhs.args[5].name)
        ):
            bodo.utils.utils.check_h5py()
            # TODO: make create_dataset/create_group collective
            arr = rhs.args[5]
            # dataset dimensions can be different than array due to integer selection
            ndims = len(self.typemap[rhs.args[2].name])
            nodes = []

            # divide 1st dimension
            size_var = self._get_dist_var_len(arr, nodes, equiv_set, avail_vars)
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

        # Adjust array index variable to be within current processor's data chunk
        # See docstring of _is_array_access_stmt
        if fdef == (
            "get_split_view_index",
            "bodo.hiframes.split_impl",
        ) and self._dist_arr_needs_adjust(rhs.args[0].name, rhs.args[1].name):
            arr = rhs.args[0]
            old_ind = rhs.args[1]
            index_var = self._fix_index_var(old_ind)
            start_var, nodes = self._get_parallel_access_start_var(
                arr, equiv_set, index_var, avail_vars
            )
            sub_nodes = self._get_ind_sub(index_var, start_var)
            out = nodes + sub_nodes
            rhs.args[1] = sub_nodes[-1].target
            if isinstance(self.typemap[old_ind.name], types.Literal):
                self._set_ith_arg_to_unliteral(rhs, 1)
            out.append(assign)
            return out

        # Adjust array index variable to be within current processor's data chunk
        # See docstring of _is_array_access_stmt
        if fdef == (
            "setitem_str_arr_ptr",
            "bodo.libs.str_arr_ext",
        ) and self._dist_arr_needs_adjust(rhs.args[0].name, rhs.args[1].name):
            arr = rhs.args[0]
            old_ind = rhs.args[1]
            index_var = self._fix_index_var(old_ind)
            start_var, nodes = self._get_parallel_access_start_var(
                arr, equiv_set, index_var, avail_vars
            )
            sub_nodes = self._get_ind_sub(index_var, start_var)
            out = nodes + sub_nodes
            rhs.args[1] = sub_nodes[-1].target
            if isinstance(self.typemap[old_ind.name], types.Literal):
                self._set_ith_arg_to_unliteral(rhs, 1)
            out.append(assign)
            return out

        # Adjust array index variable to be within current processor's data chunk
        # See docstring of _is_array_access_stmt
        if fdef in (
            (
                "inplace_eq",
                "bodo.libs.str_arr_ext",
            ),
            ("str_arr_setitem_int_to_str", "bodo.libs.str_arr_ext"),
            ("str_arr_setitem_NA_str", "bodo.libs.str_arr_ext"),
            ("str_arr_set_not_na", "bodo.libs.str_arr_ext"),
        ) and self._dist_arr_needs_adjust(rhs.args[0].name, rhs.args[1].name):
            arr = rhs.args[0]
            old_ind = rhs.args[1]
            index_var = self._fix_index_var(old_ind)
            start_var, nodes = self._get_parallel_access_start_var(
                arr, equiv_set, index_var, avail_vars
            )
            sub_nodes = self._get_ind_sub(index_var, start_var)
            out = nodes + sub_nodes
            rhs.args[1] = sub_nodes[-1].target
            if isinstance(self.typemap[old_ind.name], types.Literal):
                self._set_ith_arg_to_unliteral(rhs, 1)
            out.append(assign)
            return out

        # Adjust array index variable to be within current processor's data chunk
        # See docstring of _is_array_access_stmt
        if fdef == (
            "str_arr_item_to_numeric",
            "bodo.libs.str_arr_ext",
        ):
            out = []
            # output array
            if self._dist_arr_needs_adjust(rhs.args[0].name, rhs.args[1].name):
                arr = rhs.args[0]
                old_ind = rhs.args[1]
                index_var = self._fix_index_var(old_ind)
                start_var, nodes = self._get_parallel_access_start_var(
                    arr, equiv_set, index_var, avail_vars
                )
                sub_nodes = self._get_ind_sub(index_var, start_var)
                out += nodes + sub_nodes
                rhs.args[1] = sub_nodes[-1].target
                if isinstance(self.typemap[old_ind.name], types.Literal):
                    self._set_ith_arg_to_unliteral(rhs, 1)
            # input string array
            if self._dist_arr_needs_adjust(rhs.args[2].name, rhs.args[3].name):
                arr = rhs.args[2]
                old_ind = rhs.args[3]
                index_var = self._fix_index_var(old_ind)
                start_var, nodes = self._get_parallel_access_start_var(
                    arr, equiv_set, index_var, avail_vars
                )
                sub_nodes = self._get_ind_sub(index_var, start_var)
                out += nodes + sub_nodes
                rhs.args[3] = sub_nodes[-1].target
                if isinstance(self.typemap[old_ind.name], types.Literal):
                    self._set_ith_arg_to_unliteral(rhs, 3)
            out.append(assign)
            return out

        # Adjust array index variable to be within current processor's data chunk
        # See docstring of _is_array_access_stmt
        if fdef in (
            (
                "get_str_arr_item_copy",
                "bodo.libs.str_arr_ext",
            ),
            ("copy_array_element", "bodo.libs.array_kernels"),
        ):
            out = []
            # output string array
            if self._dist_arr_needs_adjust(rhs.args[0].name, rhs.args[1].name):
                arr = rhs.args[0]
                old_ind = rhs.args[1]
                index_var = self._fix_index_var(old_ind)
                start_var, nodes = self._get_parallel_access_start_var(
                    arr, equiv_set, index_var, avail_vars
                )
                sub_nodes = self._get_ind_sub(index_var, start_var)
                out += nodes + sub_nodes
                rhs.args[1] = sub_nodes[-1].target
                if isinstance(self.typemap[old_ind.name], types.Literal):
                    self._set_ith_arg_to_unliteral(rhs, 1)

            # input string array
            if self._dist_arr_needs_adjust(rhs.args[2].name, rhs.args[3].name):
                arr = rhs.args[2]
                old_ind = rhs.args[3]
                index_var = self._fix_index_var(old_ind)
                start_var, nodes = self._get_parallel_access_start_var(
                    arr, equiv_set, index_var, avail_vars
                )
                sub_nodes = self._get_ind_sub(index_var, start_var)
                out += nodes + sub_nodes
                rhs.args[3] = sub_nodes[-1].target
                if isinstance(self.typemap[old_ind.name], types.Literal):
                    self._set_ith_arg_to_unliteral(rhs, 3)

            out.append(assign)
            return out

        # Adjust array index variable to be within current processor's data chunk
        # See docstring of _is_array_access_stmt
        if fdef == ("setna", "bodo.libs.array_kernels") and self._dist_arr_needs_adjust(
            rhs.args[0].name, rhs.args[1].name
        ):
            arr = rhs.args[0]
            old_ind = rhs.args[1]
            index_var = self._fix_index_var(old_ind)
            start_var, nodes = self._get_parallel_access_start_var(
                arr, equiv_set, index_var, avail_vars
            )
            sub_nodes = self._get_ind_sub(index_var, start_var)
            out = nodes + sub_nodes
            rhs.args[1] = sub_nodes[-1].target
            if isinstance(self.typemap[old_ind.name], types.Literal):
                self._set_ith_arg_to_unliteral(rhs, 1)
            out.append(assign)
            return out

        # Adjust array index variable to be within current processor's data chunk
        # See docstring of _is_array_access_stmt
        if fdef in (
            ("isna", "bodo.libs.array_kernels"),
            ("get_bit_bitmap_arr", "bodo.libs.int_arr_ext"),
            ("set_bit_to_arr", "bodo.libs.int_arr_ext"),
            ("get_str_arr_str_length", "bodo.libs.str_arr_ext"),
            ("scalar_optional_getitem", "bodo.utils.indexing"),
        ) and self._dist_arr_needs_adjust(rhs.args[0].name, rhs.args[1].name):
            # fix index in call to isna
            arr = rhs.args[0]
            old_ind = rhs.args[1]
            ind = self._fix_index_var(old_ind)
            start_var, out = self._get_parallel_access_start_var(
                arr, equiv_set, ind, avail_vars
            )
            out += self._get_ind_sub(ind, start_var)
            rhs.args[1] = out[-1].target
            if isinstance(self.typemap[old_ind.name], types.Literal):
                self._set_ith_arg_to_unliteral(rhs, 1)
            out.append(assign)

        if fdef in (
            (
                "rolling_fixed",
                "bodo.hiframes.rolling",
            ),
            (
                "rolling_variable",
                "bodo.hiframes.rolling",
            ),
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            set_last_arg_to_true(self, assign.value)
            return [assign]

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

        # Note for both of these functions:
        # Case 1: DIST DIST -> DIST, is_parallel=True
        # Case 2: REP  REP  -> REP, is_parallel=False
        # Case 3: DIST REP  -> DIST, is_parallel=False
        # Case 4: REP  DIST:   Banned by construction
        if fdef == ("array_isin", "bodo.libs.array") and self._is_1D_or_1D_Var_arr(
            rhs.args[2].name
        ):
            # array_isin requires shuffling data only if values array is distributed
            f = eval(
                "lambda out_arr, in_arr, vals, p: bodo.libs.array.array_isin("
                "    out_arr, in_arr, vals, True"
                ")"
            )
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == (
            "is_in",
            "bodosql.kernels",
        ) and self._is_1D_or_1D_Var_arr(rhs.args[1].name):
            set_last_arg_to_true(self, assign.value)
            return

        if (
            fdef[0]
            in {
                "anyvalue_agg",
                "boolor_agg",
                "booland_agg",
                "boolxor_agg",
                "bitor_agg",
                "bitand_agg",
                "bitxor_agg",
            }
            and fdef[1] == "bodo.libs.array_kernels"
            and self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            arr = rhs.args[0]
            f = eval(f"lambda A: bodo.libs.array_kernels.{fdef[0]}(A, True)")
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == (
            "quantile",
            "bodo.libs.array_kernels",
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            arr = rhs.args[0]
            nodes = []
            size_var = self._get_dist_var_len(arr, nodes, equiv_set, avail_vars)
            rhs.args.append(size_var)

            f = eval(
                "lambda arr, q, size: bodo.libs.array_kernels.quantile_parallel("
                "    arr, q, size"
                ")"
            )
            return nodes + compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef in (
            ("approx_percentile", "bodo.libs.array_kernels"),
            ("percentile_cont", "bodo.libs.array_kernels"),
            ("percentile_disc", "bodo.libs.array_kernels"),
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == ("nunique", "bodo.libs.array_kernels") and self._is_1D_or_1D_Var_arr(
            rhs.args[0].name
        ):
            f = eval(
                "lambda arr, dropna: bodo.libs.array_kernels.nunique_parallel(arr, dropna)"
            )
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("unique", "bodo.libs.array_kernels") and self._is_1D_or_1D_Var_arr(
            rhs.args[0].name
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == (
            "accum_func",
            "bodo.libs.array_kernels",
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == (
            "intersection_mask",
            "bodo.libs.array_kernels",
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == (
            "first_last_valid_index",
            "bodo.libs.array_kernels",
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == (
            "get_valid_entries_from_date_offset",
            "bodo.libs.array_kernels",
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == ("pivot_impl", "bodo.hiframes.pd_dataframe_ext") and (
            self._is_1D_tup(rhs.args[0].name) or self._is_1D_Var_tup(rhs.args[0].name)
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == (
            "ffill_bfill_arr",
            "bodo.libs.array_kernels",
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == ("nonzero", "bodo.libs.array_kernels") and self._is_1D_or_1D_Var_arr(
            rhs.args[0].name
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == (
            "nlargest",
            "bodo.libs.array_kernels",
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            f = eval(
                "lambda arr, I, k, i, f: bodo.libs.array_kernels.nlargest_parallel("
                "    arr, I, k, i, f"
                ")"
            )
            return compile_func_single_block(f, rhs.args, assign.target, self)

        if fdef == ("nancorr", "bodo.libs.array_kernels") and (
            self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == ("series_monotonicity", "bodo.libs.array_kernels") and (
            self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == ("autocorr", "bodo.libs.array_kernels") and (
            self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == ("array_op_median", "bodo.libs.array_ops") and (
            self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == ("str_arr_min_max", "bodo.libs.str_arr_ext") and (
            self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == ("array_op_describe", "bodo.libs.array_ops") and (
            self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            # If describe is parallel, replace with a bodo compiled implementation
            # to handle parallelism.
            import bodo.libs.parallel_ops as parallel_ops

            impl = parallel_ops.get_array_op_describe_dispatcher(
                self.typemap[rhs.args[0].name]
            )
            return compile_func_single_block(
                eval("lambda arr: f(arr)"),
                (rhs.args[0],),
                assign.target,
                self,
                extra_globals={"bodo.libs.parallel_ops": parallel_ops, "f": impl},
            )

        if fdef == ("array_op_nbytes", "bodo.libs.array_ops") and (
            self._is_1D_or_1D_Var_arr(rhs.args[0].name)
        ):
            # If array_op_nbytes is parallel, replace with a bodo compiled implementation
            # to handle parallelism.
            import bodo.libs.parallel_ops as parallel_ops

            impl = parallel_ops.array_op_nbytes_parallel
            return compile_func_single_block(
                eval("lambda arr: f(arr)"),
                (rhs.args[0],),
                assign.target,
                self,
                extra_globals={"bodo.libs.parallel_ops": parallel_ops, "f": impl},
            )

        if fdef == ("duplicated", "bodo.libs.array_kernels") and (
            self._is_1D_tup(rhs.args[0].name) or self._is_1D_Var_tup(rhs.args[0].name)
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == ("drop_duplicates", "bodo.libs.array_kernels") and (
            self._is_1D_tup(rhs.args[0].name) or self._is_1D_Var_tup(rhs.args[0].name)
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == (
            "drop_duplicates_table",
            "bodo.utils.table_utils",
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == ("union_tables", "bodo.libs.array") and self._is_1D_or_1D_Var_arr(
            lhs
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == (
            "drop_duplicates_array",
            "bodo.libs.array_kernels",
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if func_name == "rebalance" and func_mod in {
            "bodo.libs.distributed_api",
            "bodo",
        }:
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                set_last_arg_to_true(self, assign.value)
                return [assign]
            else:
                warnings.warn("Invoking rebalance on a replicated array has no effect")

        if fdef == (
            "get_chunk_bounds",
            "bodo.libs.distributed_api",
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if func_name == "random_shuffle" and func_mod in {
            "bodo.libs.distributed_api",
            "bodo",
        }:
            if self._is_1D_or_1D_Var_arr(rhs.args[0].name):
                set_last_arg_to_true(self, assign.value)
                return [assign]

        if fdef == ("sample_table_operation", "bodo.libs.array_kernels") and (
            self._is_1D_tup(rhs.args[0].name) or self._is_1D_Var_tup(rhs.args[0].name)
        ):
            set_last_arg_to_true(self, assign.value)
            return [assign]

        if fdef == (
            "init_range_index",
            "bodo.hiframes.pd_index_ext",
        ) and self._is_1D_or_1D_Var_arr(lhs):
            return self._run_call_init_range_index(
                lhs, assign, rhs.args, avail_vars, equiv_set
            )

        if fdef == (
            "generate_empty_table_with_rows",
            "bodo.hiframes.table",
        ) and self._is_1D_arr(lhs):
            return self._run_call_generate_empty_table_with_rows(
                lhs, assign, rhs.args, avail_vars, equiv_set
            )

        if fdef == ("_bodo_groupby_apply_impl", "") and self._is_1D_or_1D_Var_arr(lhs):
            # inline shuffling of groupby apply data to make sure input arrays are
            # deallocated during shuffle and there is less memory pressure.
            # inlining causes dels to be generated earlier for variables than calling a
            # the _bodo_groupby_apply_impl function
            # see [BE-2079]
            keys = rhs.args[0]
            in_df = rhs.args[1]
            keys_type = self.typemap[keys.name]
            in_df_type = self.typemap[in_df.name]

            # generate shuffle_dataframe(in_df, keys, True)
            py_func, glbls = bodo.hiframes.pd_groupby_ext.gen_shuffle_dataframe(
                in_df_type, keys_type, types.literal(True)
            )
            true_var = ir.Var(assign.target.scope, mk_unique_var("true"), rhs.loc)
            self.typemap[true_var.name] = types.bool_
            nodes = [ir.Assign(ir.Const(True, rhs.loc), true_var, rhs.loc)]
            nodes += compile_func_single_block(
                py_func, [in_df, keys, true_var], None, self, extra_globals=glbls
            )
            out_tup = nodes[-1].value
            out_df = out_tup.items[0]
            out_keys = out_tup.items[1]
            shuffle_info = out_tup.items[2]

            # update _bodo_groupby_apply_impl() call to use shuffled data
            rhs.args[0] = out_keys
            rhs.args[1] = out_df
            # shuffle info and is_parallel are last arguments
            rhs.args[-2] = shuffle_info
            rhs.args[-1] = true_var
            nodes.append(assign)
            return nodes

        # no need to gather if input data is replicated
        if (
            (fdef == ("gatherv", "bodo") or fdef == ("allgatherv", "bodo"))
            and self._is_REP(rhs.args[0].name)
            and self.typemap[rhs.args[0].name] == self.typemap[assign.target.name]
        ):
            # NOTE: input/output type match check is necessary for dictionary encoded
            # string array since it is converted to a regular string array
            assign.value = rhs.args[0]
            return [assign]

        # no need to gather if input data is replicated, but the data is a
        # readonly array we need to make a copy.
        if (
            (fdef == ("gatherv", "bodo") or fdef == ("allgatherv", "bodo"))
            and self._is_REP(rhs.args[0].name)
            # TODO: Can other types except arrays be read only.
            and isinstance(self.typemap[rhs.args[0].name], types.Array)
            and isinstance(self.typemap[assign.target.name], types.Array)
        ):
            modifiable_input = self.typemap[rhs.args[0].name].copy(readonly=False)
            modifiable_output = self.typemap[assign.target.name].copy(readonly=False)
            if modifiable_input == modifiable_output:
                # If data is equal make a copy.
                return compile_func_single_block(
                    eval("lambda data: data.copy()"),
                    (rhs.args[0],),
                    assign.target,
                    self,
                )

        # if input data is replicated and we convert from dict_array to string
        # array, replace with decode_if_dict_array (which happens inside gatherv).
        if (
            (fdef == ("gatherv", "bodo") or fdef == ("allgatherv", "bodo"))
            and self._is_REP(rhs.args[0].name)
            # Was data converted from dict -> str inside gatherv.
            and to_str_arr_if_dict_array(self.typemap[rhs.args[0].name])
            == to_str_arr_if_dict_array(self.typemap[assign.target.name])
        ):
            return compile_func_single_block(
                eval("lambda data: decode_if_dict_array(data)"),
                (rhs.args[0],),
                assign.target,
                self,
                extra_globals={"decode_if_dict_array": decode_if_dict_array},
            )

        # no need to scatter if input data is distributed
        if (fdef == ("scatterv", "bodo")) and self._is_1D_or_1D_Var_arr(
            rhs.args[0].name
        ):
            lhs_typ = self.typemap[assign.target.name]
            rhs_typ = self.typemap[rhs.args[0].name]
            if lhs_typ != rhs_typ and to_str_arr_if_dict_array(
                lhs_typ
            ) == to_str_arr_if_dict_array(rhs_typ):
                # If the difference is table format we decode arrays as opposed to depending
                # on a cast because cast isn't implemented yet.
                return compile_func_single_block(
                    eval("lambda data: decode_if_dict_array(data)"),
                    (rhs.args[0],),
                    assign.target,
                    self,
                    extra_globals={"decode_if_dict_array": decode_if_dict_array},
                )

            else:
                # We also need to optimize out scatterv and depend on casting
                # to handle any type mismatches (e.g. with/without table format)
                assign.value = rhs.args[0]
                return [assign]

        if fdef == ("dist_return", "bodo.libs.distributed_api"):
            assign.value = rhs.args[0]
            return [assign]

        if fdef == ("rep_return", "bodo.libs.distributed_api"):
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

            func_text = (
                ""
                "def impl(fname, data_ptr, start, count, offset):\n"
                "    return bodo.io.np_io.file_read_parallel(\n"
                "        fname, data_ptr, start, count, offset\n"
                "    )\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return nodes + compile_func_single_block(
                # Increment start_var by the file offset
                loc_vars["impl"],
                [fname, arr, start_var, count_var, file_offset],
                assign.target,
                self,
            )

        # Iceberg Merge Into
        if fdef == ("iceberg_merge_cow_py", "bodo.io.iceberg.merge_into"):
            # Dataframe is the 3rd argument (counting from 0)
            df_arg = rhs.args[3].name
            if not self._is_1D_or_1D_Var_arr(df_arg):
                raise BodoError(
                    "Merge Into with Iceberg Tables are only supported on distributed DataFrames"
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
                    eval(
                        "lambda: bodo.hiframes.pd_timestamp_ext.integer_to_dt64("
                        "    numba.cpython.builtins.get_type_max_value("
                        "        numba.core.types.uint64"
                        "    )"
                        ")"
                    ),
                    {"bodo": bodo, "numba": numba},
                    typingctx=self.typingctx,
                    targetctx=self.targetctx,
                    arg_typs=(),
                    typemap=self.typemap,
                    calltypes=self.calltypes,
                ).blocks.popitem()[1]
                out = f_block.body[:-2]
                out[-1].target = assign.target

        if fdef == ("get_type_min_value", "numba.cpython.builtins"):
            if self.typemap[rhs.args[0].name] == types.DType(types.NPDatetime("ns")):
                f_block = compile_to_numba_ir(
                    eval(
                        "lambda: bodo.hiframes.pd_timestamp_ext.integer_to_dt64("
                        "    numba.cpython.builtins.get_type_min_value("
                        "        numba.core.types.uint64"
                        "    )"
                        ")"
                    ),
                    {"bodo": bodo, "numba": numba},
                    typingctx=self.typingctx,
                    targetctx=self.targetctx,
                    arg_typs=(),
                    typemap=self.typemap,
                    calltypes=self.calltypes,
                ).blocks.popitem()[1]
                out = f_block.body[:-2]
                out[-1].target = assign.target
        if (
            fdef == ("fft2", "scipy.fftpack._basic")
            or fdef == ("fft2", "scipy.fft._basic")
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):  # pragma: no cover
            set_last_arg_to_true(self, assign.value)
            return [assign]
        if (
            fdef == ("fftshift", "numpy.fft")
            or fdef == ("fftshift", "scipy.fft._helper")
        ) and self._is_1D_or_1D_Var_arr(rhs.args[0].name):  # pragma: no cover
            set_last_arg_to_true(self, assign.value)
            return [assign]

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
            # If step = 0, evaluates as max(0, -(-(stop - start) // 1)) to
            # prevent a ZeroDivisionError from being raised before init_range_index
            # can raise its own error.
            f = eval(
                "lambda start, stop, step: max(0, -(-(stop - start) // (step + (step == 0))))"
            )
            out += compile_func_single_block(f, args[:-1], None, self)
            size_var = out[-1].target

        if self._is_1D_arr(lhs):
            start_var = self._get_1D_start(size_var, avail_vars, out)
            end_var = self._get_1D_end(size_var, out)
            self._update_avail_vars(avail_vars, out)

            if is_simple_range:
                args[0] = start_var
                args[1] = end_var

                func_text = (
                    ""
                    "def impl(start, stop, step, name):\n"
                    "    res = bodo.hiframes.pd_index_ext.init_range_index(\n"
                    "        start, stop, step, name\n"
                    "    )\n"
                    "    return res\n"
                )

            else:
                func_text = (
                    ""
                    "def impl(start, stop, step, name, chunk_start, chunk_end):\n"
                    "    chunk_start = start + step * chunk_start\n"
                    "    chunk_end = start + step * chunk_end\n"
                    "    res = bodo.hiframes.pd_index_ext.init_range_index(\n"
                    "        chunk_start, chunk_end, step, name\n"
                    "    )\n"
                    "    return res\n"
                )

                args = args + [start_var, end_var]

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return out + compile_func_single_block(
                loc_vars["impl"], args, assign.target, self
            )
        else:
            # 1D_Var case
            assert self._is_1D_Var_arr(lhs)
            assert is_simple_range, "only simple 1D_Var RangeIndex is supported"
            new_size_var = self._get_1D_Var_size(size_var, equiv_set, avail_vars, out)

            func_text = (
                ""
                "def impl(stop, name):\n"
                "    prefix = bodo.libs.distributed_api.dist_exscan(stop, _op)\n"
                "    return bodo.hiframes.pd_index_ext.init_range_index(\n"
                "        prefix, prefix + stop, 1, name\n"
                "    )\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return out + compile_func_single_block(
                loc_vars["impl"],
                [new_size_var, args[3]],
                assign.target,
                self,
                extra_globals={"_op": np.int32(Reduce_Type.Sum.value)},
            )

    def _run_call_generate_empty_table_with_rows(
        self, lhs, assign, args, avail_vars, equiv_set
    ):
        """transform generate_empty_table_with_rows() calls"""
        assert len(args) == 1, "invalid generate_empty_table_with_rows() call"
        size_var = args[0]
        out = []
        args[0] = self._get_1D_count(size_var, out)
        func_text = (
            ""
            "def impl(n_rows):\n"
            "    res = bodo.hiframes.table.generate_empty_table_with_rows(n_rows)\n"
            "    return res\n"
        )
        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        return out + compile_func_single_block(
            loc_vars["impl"], args, assign.target, self
        )

    def _run_call_np(self, lhs, func_name, assign, args, kws, equiv_set):
        """transform np.func() calls"""
        # allocs are handled separately
        assert not (
            self._is_1D_or_1D_Var_arr(lhs)
            and func_name in bodo.utils.utils.np_alloc_callnames
        ), "allocation calls handled separately 'empty', 'zeros', 'ones', 'full' etc."
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
                shape_vars = find_build_tuple(self.func_ir, args[1], True)
            return self._run_np_reshape(assign, args[0], shape_vars, equiv_set)

        if func_name in list_cumulative and self._is_1D_or_1D_Var_arr(args[0].name):
            in_arr_var = args[0]
            lhs_var = assign.target
            # TODO: compute inplace if input array is dead
            func_text = (
                ""
                "def impl(A):\n"
                "    B = np.empty_like(A)\n"
                "    _func(A, B)\n"
                "    return B\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            func = getattr(bodo.libs.distributed_api, "dist_" + func_name)
            return compile_func_single_block(
                loc_vars["impl"],
                [in_arr_var],
                lhs_var,
                self,
                extra_globals={"_func": func},
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
                shape_vars = find_build_tuple(self.func_ir, args[0], True)
            return self._run_np_reshape(assign, arr, shape_vars, equiv_set)

        # TODO: refactor
        # TODO: add unittest
        if func_name == "tofile":
            if self._is_1D_arr(arr.name):
                _fname = args[0]
                nodes, start_var, count_var = self._get_dist_var_start_count(
                    arr, equiv_set, avail_vars
                )

                func_text = (
                    ""
                    "def f(fname, arr, start, count):\n"
                    "    return bodo.io.np_io.file_write_parallel(fname, arr, start, count)\n"
                )

                loc_vars = {}
                exec(func_text, globals(), loc_vars)
                return nodes + compile_func_single_block(
                    loc_vars["f"],
                    [_fname, arr, start_var, count_var],
                    assign.target,
                    self,
                )

            if self._is_1D_Var_arr(arr.name):
                _fname = args[0]

                func_text = (
                    ""
                    "def f(fname, arr):\n"
                    "    count = len(arr)\n"
                    "    start = bodo.libs.distributed_api.dist_exscan(count, _op)\n"
                    "    return bodo.io.np_io.file_write_parallel(fname, arr, start, count)\n"
                )

                loc_vars = {}
                exec(func_text, globals(), loc_vars)
                return compile_func_single_block(
                    loc_vars["f"],
                    [_fname, arr],
                    assign.target,
                    self,
                    extra_globals={"_op": np.int32(Reduce_Type.Sum.value)},
                )

        return out

    def _run_call_boolean_array(self, arr, func_name, assign):
        """transform distributed BooleanArray.func calls"""
        out = [assign]
        if func_name == "all":
            if self._is_1D_or_1D_Var_arr(arr.name):
                reduce_op = Reduce_Type.Logical_And
                reduce_var = assign.target
                scope = assign.target.scope
                loc = assign.loc
                return out + self._gen_reduce(reduce_var, reduce_op, scope, loc)

        return out

    def _run_call_df(self, lhs, df, func_name, assign, args):
        """transform DataFrame calls to be distributed"""
        if func_name in ("to_parquet", "to_sql") and self._is_1D_or_1D_Var_arr(df.name):
            set_last_arg_to_true(self, assign.value)
            return [assign]
        elif func_name == "to_csv" and self._is_1D_or_1D_Var_arr(df.name):
            # avoid header for non-zero ranks
            # write to string then parallel file write
            # df.to_csv(fname, _bodo_file_prefix) ->
            # header = header and is_root  # only first line has header
            # str_out = df.to_csv(None, header=header)
            # bodo.io.csv_cpp(fname, str_out, _bodo_file_prefix)

            df_typ = self.typemap[df.name]
            rhs = assign.value
            kws = dict(rhs.kws)
            nodes = []

            fname = get_call_expr_arg(
                "to_csv",
                rhs.args,
                kws,
                0,
                "path_or_buf",
                default=None,
                use_default=True,
            )

            if "_bodo_file_prefix" in kws:
                file_prefix_var = get_call_expr_arg(
                    "to_csv",
                    rhs.args,
                    kws,
                    21,
                    "_bodo_file_prefix",
                    default="part-",
                    use_default=True,
                )

                file_prefix_val = self.typemap[file_prefix_var.name].literal_value
            else:
                file_prefix_val = "part-"

            file_prefix = ir.Var(
                assign.target.scope, mk_unique_var("file_prefix"), rhs.loc
            )
            self.typemap[file_prefix.name] = types.unicode_type
            nodes.append(
                ir.Assign(ir.Const(file_prefix_val, df.loc), file_prefix, df.loc)
            )

            # handle None filepath
            if fname is None or isinstance(self.typemap[fname.name], types.NoneType):
                return [assign]
            # convert StringLiteral to Unicode to make ._data available
            self.typemap.pop(fname.name)
            self.typemap[fname.name] = string_type

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
            set_call_expr_arg(none_var, rhs.args, kws, 0, "path_or_buf")

            # str_out = df.to_csv(None)
            str_out = ir.Var(assign.target.scope, mk_unique_var("write_csv"), rhs.loc)
            self.typemap[str_out.name] = string_type
            new_assign = ir.Assign(rhs, str_out, rhs.loc)
            nodes.append(new_assign)

            # print_node = ir.Print([str_out], None, rhs.loc)
            # self.calltypes[print_node] = signature(types.none, string_type)
            # nodes.append(print_node)

            # TODO: fix lazy IO load
            func_text = (
                ""
                "def f(fname, str_out, file_prefix):\n"
                "    utf8_str, utf8_len = unicode_to_utf8_and_len(str_out)\n"
                "    start = bodo.libs.distributed_api.dist_exscan(utf8_len, _op)\n"
                "    # Assuming that path_or_buf is a string\n"
                "    bucket_region = bodo.io.fs_io.get_s3_bucket_region_wrapper(fname, parallel=True)\n"
                "    # TODO: unicode file name\n"
                "    _csv_write(\n"
                "        unicode_to_utf8(fname),\n"
                "        utf8_str,\n"
                "        start,\n"
                "        utf8_len,\n"
                "        True,\n"
                "        unicode_to_utf8(bucket_region),\n"
                "        unicode_to_utf8(file_prefix),\n"
                "    )\n"
                "    # Check if there was an error in the C++ code. If so, raise it.\n"
                "    bodo.utils.utils.check_and_propagate_cpp_exception()\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return nodes + compile_func_single_block(
                loc_vars["f"],
                [fname, str_out, file_prefix],
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
            kws = dict(rhs.kws)
            fname = get_call_expr_arg(
                "to_json",
                rhs.args,
                kws,
                0,
                "path_or_buf",
                default=None,
                use_default=True,
            )
            if fname is None or isinstance(self.typemap[fname.name], types.NoneType):
                return [assign]
            # convert StringLiteral to Unicode to make ._data available
            self.typemap.pop(fname.name)
            self.typemap[fname.name] = string_type
            nodes = []

            is_records = False
            if "orient" in kws:
                orient_var = get_call_expr_arg(
                    "to_json", rhs.args, kws, 1, "orient", None
                )
                orient_val = self.typemap[orient_var.name]
                if not is_overload_constant_str(orient_val):
                    raise BodoError(
                        "orient argument in to_json() must be a constant string"
                    )
                is_records = (
                    True if get_overload_const_str(orient_val) == "records" else False
                )

            is_lines = False
            if "lines" in kws:
                lines_var = get_call_expr_arg(
                    "to_json", rhs.args, kws, 7, "lines", None
                )
                lines_val = self.typemap[lines_var.name]
                if not is_overload_constant_bool(lines_val):
                    raise BodoError(
                        "lines argument in to_json() must be a constant boolean"
                    )
                is_lines = get_overload_const_bool(lines_val)

            is_records_lines = ir.Var(
                assign.target.scope, mk_unique_var("is_records_lines"), rhs.loc
            )
            self.typemap[is_records_lines.name] = types.bool_
            nodes.append(
                ir.Assign(
                    ir.Const(is_records and is_lines, df.loc), is_records_lines, df.loc
                )
            )

            if "_bodo_file_prefix" in kws:
                file_prefix_var = get_call_expr_arg(
                    "to_json",
                    rhs.args,
                    kws,
                    14,
                    "_bodo_file_prefix",
                    default="part-",
                    use_default=True,
                )

                file_prefix_val = self.typemap[file_prefix_var.name].literal_value
            else:
                file_prefix_val = "part-"

            file_prefix = ir.Var(
                assign.target.scope, mk_unique_var("file_prefix"), rhs.loc
            )
            self.typemap[file_prefix.name] = types.unicode_type
            nodes.append(
                ir.Assign(ir.Const(file_prefix_val, df.loc), file_prefix, df.loc)
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
            set_call_expr_arg(none_var, rhs.args, kws, 0, "path_or_buf")
            rhs.kws = kws

            # str_out = df.to_json(None)
            str_out = ir.Var(assign.target.scope, mk_unique_var("write_json"), rhs.loc)
            self.typemap[str_out.name] = string_type
            new_assign = ir.Assign(rhs, str_out, rhs.loc)
            nodes.append(new_assign)

            # print_node = ir.Print([str_out], None, rhs.loc)
            # self.calltypes[print_node] = signature(types.none, string_type)
            # nodes.append(print_node)

            # TODO: fix lazy IO load
            func_text = (
                ""
                "def f(fname, str_out, is_records_lines, file_prefix):\n"
                "    utf8_str, utf8_len = unicode_to_utf8_and_len(str_out)\n"
                "    start = bodo.libs.distributed_api.dist_exscan(utf8_len, _op)\n"
                "    # Assuming that path_or_buf is a string\n"
                "    bucket_region = bodo.io.fs_io.get_s3_bucket_region_wrapper(fname, parallel=True)\n"
                "    # TODO: unicode file name\n"
                "    _json_write(\n"
                "        unicode_to_utf8(fname),\n"
                "        utf8_str,\n"
                "        start,\n"
                "        utf8_len,\n"
                "        True,\n"
                "        is_records_lines,\n"
                "        unicode_to_utf8(bucket_region),\n"
                "        unicode_to_utf8(file_prefix),\n"
                "    )\n"
                "    # Check if there was an error in the C++ code. If so, raise it.\n"
                "    bodo.utils.utils.check_and_propagate_cpp_exception()\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return nodes + compile_func_single_block(
                loc_vars["f"],
                [fname, str_out, is_records_lines, file_prefix],
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

    def _run_call_series(self, lhs, series, func_name, assign, args):
        if func_name == "to_csv" and self._is_1D_or_1D_Var_arr(series.name):
            set_last_arg_to_true(self, assign.value)
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

        func_text = (
            ""
            "def f(cond, fname):\n"
            "    return cond & (\n"
            "        (bodo.libs.distributed_api.get_rank() == 0)\n"
            "        | _csv_output_is_dir(fname._data)\n"
            "    )\n"
        )

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        f_block = compile_to_numba_ir(
            loc_vars["f"],
            {
                "bodo": bodo,
                "_csv_output_is_dir": _csv_output_is_dir,
            },
            typingctx=self.typingctx,
            targetctx=self.targetctx,
            arg_typs=(self.typemap[cond_var.name], self.typemap[fname_var.name]),
            typemap=self.typemap,
            calltypes=self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(f_block, [cond_var, fname_var])
        nodes = f_block.body[:-2]
        return nodes

    def _run_permutation_int(self, assign, args):
        lhs = assign.target
        n = args[0]

        func_text = (
            ""
            "def f(lhs, n):\n"
            "    bodo.libs.distributed_api.dist_permutation_int(lhs, n)\n"
        )

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        f_block = compile_to_numba_ir(
            loc_vars["f"],
            {"bodo": bodo},
            typingctx=self.typingctx,
            targetctx=self.targetctx,
            arg_typs=(self.typemap[lhs.name], types.intp),
            typemap=self.typemap,
            calltypes=self.calltypes,
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

    def _const_to_var(self, v, nodes, scope, loc):
        """Convert constant value to ir.Var if necessary.

        Args:
            v (int|ir.Var): input value (int or ir.Var)
            nodes (list(ir.Stmt)): list of generated IR nodes for appending new nodes
            scope (ir.Scope): IR scope object
            loc (ir.Loc): IR loc object

        Returns:
            ir.Var: new variable for value v
        """
        if isinstance(v, ir.Var):
            return v

        new_var = ir.Var(scope, mk_unique_var("const_var"), loc)
        self.typemap[new_var.name] = types.literal(v)
        nodes.append(ir.Assign(ir.Const(v, loc), new_var, loc))
        return new_var

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
                eval("lambda A: A.reshape(A.size)"), [in_arr], lhs, self
            )

        # optimization: no need to distribute if 1-dim array is reshaped to
        # 2-dim with same length (just added a new dimension)
        if (
            self.typemap[in_arr.name].ndim == 1
            and len(shape_vars) == 2
            and (
                (isinstance(shape_vars[1], int) and shape_vars[1] == 1)
                or (
                    guard(
                        get_const_value_inner,
                        self.func_ir,
                        shape_vars[1],
                        typemap=self.typemap,
                    )
                    == 1
                )
            )
        ):
            return compile_func_single_block(
                eval("lambda A: A.reshape(len(A), 1)"), [in_arr], lhs, self
            )

        shape_vars = [self._const_to_var(v, nodes, scope, loc) for v in shape_vars]

        # get local size for 1st dimension and allocate output array
        # shape_vars[0] is global size
        count_var = self._get_1D_count(shape_vars[0], nodes)
        dtype = self.typemap[in_arr.name].dtype
        nodes += mk_alloc(
            self.typingctx,
            self.typemap,
            self.calltypes,
            lhs,
            (count_var,) + tuple(shape_vars[1:]),
            dtype,
            scope,
            loc,
            self.typemap[lhs.name],
        )

        # shuffle the data to fill output arrays on different ranks properly
        return nodes + compile_func_single_block(
            eval(
                "lambda lhs, in_arr, new_dim0_global_len: bodo.libs.distributed_api.dist_oneD_reshape_shuffle("
                "    lhs, in_arr, new_dim0_global_len"
                ")"
            ),
            [lhs, in_arr, shape_vars[0]],
            None,
            self,
        )

    def _run_call_np_dot(self, lhs, assign, args):
        out = [assign]
        arg0 = args[0].name
        arg1 = args[1].name

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
                        eval("lambda n: (n + 7) >> 3"), (count_var,), None, self
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
            oned_varnames = {
                v
                for v in self._dist_analysis.array_dists
                if self._dist_analysis.array_dists[v] == Distribution.OneD
            }
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
            guard(find_callname, self.func_ir, size_def, self.typemap)
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
            # 'v' could be int (size value) or str (varname)
            if isinstance(v, str) and "#" in v and self._is_1D_Var_arr(v.split("#")[0]):
                arr_name = v.split("#")[0]
                if arr_name not in avail_vars:
                    continue
                arr_var = ir.Var(size_var.scope, arr_name, size_var.loc)
                out += compile_func_single_block(
                    eval("lambda A: len(A)"), (arr_var,), None, self
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
                    eval("lambda A: len(A)"), (arr_var,), None, self
                )  # pragma: no cover
                new_size_var = out[-1].target

        if new_size_var is None:
            # Series.combine() uses max(s1, s2) to get output size
            calc_call = guard(find_callname, self.func_ir, size_def, self.typemap)
            if calc_call == ("max", "builtins"):
                s1 = self._get_1D_Var_size(size_def.args[0], equiv_set, avail_vars, out)
                s2 = self._get_1D_Var_size(size_def.args[1], equiv_set, avail_vars, out)
                out += compile_func_single_block(
                    eval("lambda a1, a2: max(a1, a2)"), (s1, s2), None, self
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
                            eval(
                                "lambda I: bodo.libs.array_kernels.calc_nitems("
                                "    I._start, I._stop, I._step"
                                ")"
                            ),
                            (range_val,),
                            None,
                            self,
                        )
                        new_size_var = out[-1].target

            # k = bodo.utils.indexing.bitmap_size(n) is used for calculating
            # bitmap sizes in pd_datetime_arr
            if guard(find_callname, self.func_ir, size_def, self.typemap) == (
                "bitmap_size",
                "bodo.utils.indexing",
            ):
                size = self._get_1D_Var_size(
                    size_def.args[0], equiv_set, avail_vars, out
                )
                out += compile_func_single_block(
                    eval("lambda n: bodo.utils.indexing.bitmap_size(n)"),
                    (size,),
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
                        eval("lambda n: (n + 7) >> 3"), (size,), None, self
                    )
                    new_size_var = out[-1].target

        assert new_size_var, "1D Var size not found"
        return new_size_var

    def _run_array_shape(self, lhs, arr, equiv_set, avail_vars):
        """transform array.shape to return distributed shape"""
        ndims = self.typemap[arr.name].ndim

        nodes = []
        size_var = self._get_dist_var_len(arr, nodes, equiv_set, avail_vars)
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
                eval("lambda A, size_var: (size_var,)"), (arr, size_var), lhs, self
            )
        else:
            return nodes + compile_func_single_block(
                eval("lambda A, size_var: (size_var,) + A.shape[1:]"),
                (arr, size_var),
                lhs,
                self,
            )

    def _run_array_size(self, lhs, arr, equiv_set, avail_vars):
        """transform array.size to return distributed size"""
        # get total size by multiplying all dimension sizes
        nodes = []
        if self._is_1D_arr(arr.name):
            dim1_size = self._get_dist_var_len(arr, nodes, equiv_set, avail_vars)
        else:
            assert self._is_1D_Var_arr(arr.name)
            nodes += self._gen_1D_Var_len(arr)
            dim1_size = nodes[-1].target

        func_text = (
            ""
            "def f(arr, dim1):\n"
            "    sizes = np.array(arr.shape)\n"
            "    sizes[0] = dim1\n"
            "    s = sizes.prod()\n"
        )

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        f_ir = compile_to_numba_ir(
            loc_vars["f"],
            {"np": np},
            typingctx=self.typingctx,
            targetctx=self.targetctx,
            arg_typs=(self.typemap[arr.name], types.intp),
            typemap=self.typemap,
            calltypes=self.calltypes,
        )
        f_block = f_ir.blocks.popitem()[1]
        replace_arg_nodes(f_block, [arr, dim1_size])
        nodes += f_block.body[:-3]
        nodes[-1].target = lhs
        return nodes

    def _run_array_transpose(self, assign, arr):
        """transform array.T to distributed (requires all-to-all data redistribution)"""
        # Distributed transpose is not required if output is only used in np.dot()
        # with reduction ("np.dot(X.T,Y)" pattern).
        # See np.dot() handling in distributed analysis.
        # TODO(ehsan): use DU-chain or other standard compiler infrastructure
        only_dot_transpose_reduce = True
        for block in self.func_ir.blocks.values():
            for inst in block.body:
                if inst is assign:
                    continue
                if (
                    guard(self._is_dot_transpose_reduce, inst, arr)
                    and assign.target.name == inst.value.args[0].name
                ):
                    continue
                if any(v.name == assign.target.name for v in inst.list_vars()):
                    only_dot_transpose_reduce = False
                    break

        if only_dot_transpose_reduce:
            return [assign]

        return compile_func_single_block(
            eval("lambda arr: bodo.libs.distributed_api.distributed_transpose(arr)"),
            [arr],
            assign.target,
            self,
        )

    def _is_dot_transpose_reduce(self, inst, arr):
        """Return True if statement 'inst' has the form np.dot(X.T,Y) where X is 'arr'.
        Returns False or raises GuardException if not.
        See np.dot() handling in distributed analysis.
        """
        require(is_call_assign(inst))
        require(find_callname(self.func_ir, inst.value) == ("dot", "numpy"))
        arg0 = inst.value.args[0].name
        arg1 = inst.value.args[1].name
        ndim0 = self.typemap[arg0].ndim
        ndim1 = self.typemap[arg1].ndim
        t0 = guard(_is_transposed_array, self.func_ir, arg0)
        t1 = guard(_is_transposed_array, self.func_ir, arg1)
        require(ndim0 == 2 and ndim1 == 2 and t0 and not t1)
        arg0_def = get_definition(self.func_ir, arg0)
        require(is_expr(arg0_def, "getattr") and arg0_def.attr == "T")
        return arg0_def.value.name == arr.name

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

    def _run_getitem_scalar_optional(self, full_node, equiv_set, avail_vars):
        """Transform distributed getitem/setitem operations"""
        out = [full_node]
        lhs = full_node.target
        rhs = full_node.value
        arr = rhs.args[0]
        index_var = rhs.args[1]

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
            start_var, nodes = self._get_parallel_access_start_var(
                arr, equiv_set, index_var, avail_vars
            )
            sub_nodes = self._get_ind_sub(index_var, start_var)
            out = nodes + sub_nodes
            # Update the index with the modified index.
            rhs.args[1] = sub_nodes[-1].target
            # Update the calltypes if the index was a literal.
            if isinstance(self.typemap[index_var.name], types.Literal):
                self._set_ith_arg_to_unliteral(rhs, 1)
            out.append(full_node)
        # parallel access in 1D_Var case, no need to transform
        elif (arr.name, index_var.name) in self._parallel_accesses:
            return out
        elif self._is_1D_or_1D_Var_arr(arr.name):
            start_var, nodes = self._get_dist_start_var(arr, equiv_set, avail_vars)
            size_var = self._get_dist_var_len(arr, nodes, equiv_set, avail_vars)
            is_1D = self._is_1D_arr(arr.name)
            return nodes + compile_func_single_block(
                eval(
                    "lambda arr, ind, start, tot_len: bodo.libs.distributed_api.int_optional_getitem("
                    "    arr, ind, start, tot_len, _is_1D"
                    ")"
                ),
                [arr, index_var, start_var, size_var],
                lhs,
                self,
                extra_globals={"_is_1D": is_1D},
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

            func_text = (
                ""
                "def f(A, start, step):\n"
                "    offset = abs(step - (start % step)) % step\n"
                "    return A[offset::step]\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            out += compile_func_single_block(
                loc_vars["f"], [in_arr, start_var, step], None, self
            )
            out[-1].target = lhs

        # general slice access like A[3:7]
        elif isinstance(index_typ, types.SliceType):
            in_arr = full_node.value.value
            start_var, nodes = self._get_dist_start_var(in_arr, equiv_set, avail_vars)
            size_var = self._get_dist_var_len(in_arr, nodes, equiv_set, avail_vars)
            # for multi-dim case, perform selection in other dimensions then handle
            # the first dimension
            if is_multi_dim:
                # gen index with first dimension as full slice, other dimensions as
                # full getitem index
                nodes += compile_func_single_block(
                    eval("lambda ind: (slice(None),) + ind[1:]"),
                    [full_index_var],
                    None,
                    self,
                )
                other_ind = nodes[-1].target
                return nodes + compile_func_single_block(
                    eval(
                        "lambda arr, slice_index, start, tot_len, other_ind: bodo.libs.distributed_api.slice_getitem("
                        "    operator.getitem(arr, other_ind),"
                        "    slice_index,"
                        "    start,"
                        "    tot_len,"
                        ")"
                    ),
                    [in_arr, index_var, start_var, size_var, other_ind],
                    lhs,
                    self,
                    extra_globals={"operator": operator},
                )
            return nodes + compile_func_single_block(
                eval(
                    "lambda arr, slice_index, start, tot_len: bodo.libs.distributed_api.slice_getitem("
                    "    arr,"
                    "    slice_index,"
                    "    start,"
                    "    tot_len,"
                    ")"
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
            size_var = self._get_dist_var_len(in_arr, nodes, equiv_set, avail_vars)
            is_1D = self._is_1D_arr(arr.name)
            return nodes + compile_func_single_block(
                eval(
                    "lambda arr, ind, start, tot_len: bodo.libs.distributed_api.int_getitem("
                    "    arr, ind, start, tot_len, _is_1D"
                    ")"
                ),
                [in_arr, orig_index_var, start_var, size_var],
                lhs,
                self,
                extra_globals={"_is_1D": is_1D},
            )
        # Tracing here disabled for now (https://bodo.atlassian.net/browse/BE-1213)
        #        # generate performance tracing event for distributed array filtering
        #        # e.g. ev = Event("filter ..."); A2 = A1[b]; ev.finalize()
        #        elif is_list_like_index_type(index_typ) and index_typ.dtype == types.bool_:
        #            event_nodes = self._gen_start_event("filter")
        #            ev_var = event_nodes[-1].target
        #            ev_add_attr_nodes = []
        #            ev_add_attr_nodes += self._gen_event_add_attribute(
        #                ev_var, "dtype", f"{self.typemap[lhs.name].dtype}"
        #            )
        #            ev_add_attr_nodes += self._gen_event_add_attribute(
        #                ev_var, "lhs", f"{lhs.name}"
        #            )
        #            ev_add_attr_nodes += self._gen_event_add_attribute(
        #                ev_var, "rhs", f"{node.value.name}[{index_var.name}]"
        #            )
        #            finalize_nodes = self._gen_finalize_event(ev_var)
        #            return event_nodes + ev_add_attr_nodes + out + finalize_nodes

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
            arr_len = self._get_dist_var_len(arr, nodes, equiv_set, avail_vars)

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
            func_text = (
                ""
                "def f(A, val, idx, other_inds, chunk_start, arr_len):\n"
                "    new_slice = bodo.libs.distributed_api.get_local_slice(\n"
                "        idx, chunk_start, arr_len\n"
                "    )\n"
                "    new_ind = (new_slice,) + other_inds\n"
                "    # avoid tuple index for cases like Series that don't support it\n"
                "    new_ind = bodo.utils.indexing.untuple_if_one_tuple(new_ind)\n"
                "    A[new_ind] = val\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return nodes + compile_func_single_block(
                loc_vars["f"],
                [arr, node.value, index_var, other_inds_var, start_var, arr_len],
                None,
                self,
            )

        elif isinstance(index_typ, types.Integer):
            start_var, nodes = self._get_dist_start_var(arr, equiv_set, avail_vars)

            func_text = (
                ""
                "def f(A, val, index, chunk_start):\n"
                "    bodo.libs.distributed_api._set_if_in_range(A, val, index, chunk_start)\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            return nodes + compile_func_single_block(
                loc_vars["f"], [arr, node.value, index_var, start_var], None, self
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

        # Tracing here disabled for now (https://bodo.atlassian.net/browse/BE-1213)
        #        # generate performance trace event
        #        event_nodes = self._gen_start_event("Parfor")
        #        ev_var = event_nodes[-1].target
        #        ev_add_attr_nodes = []
        #        ev_add_attr_nodes += self._gen_event_add_attribute(
        #            ev_var, "parfor_ID", str(parfor.id)
        #        )
        #        ev_add_attr_nodes += self._gen_event_add_attribute(ev_var, "distribution", "1D")
        #        finalize_nodes = self._gen_finalize_event(ev_var)
        #        return event_nodes + ev_add_attr_nodes + out + finalize_nodes

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
        for arr, index, is_bitwise in array_accesses:
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
                    eval("lambda A: len(A)"), (arr_var,), None, self
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

        for arr, index, _ in array_accesses:
            assert not guard(_is_transposed_array, self.func_ir, arr), (
                "1D_Var parfor for transposed parallel array not supported"
            )

        # see if parfor index is used in compute other than array access
        # (e.g. argmin)
        l_nest = parfor.loop_nests[0]
        ind_varname = l_nest.index_variable.name
        ind_varnames = {ind_varname}
        ind_used = False

        # traverse parfor body in topo order to find parfor index copies in ind_varnames
        # before their use
        with numba.parfors.parfor.dummy_return_in_loop_body(parfor.loop_body):
            body_labels = find_topo_order(parfor.loop_body)

        for l in body_labels:
            block = parfor.loop_body[l]
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
                if not self._is_array_access_stmt(stmt) and ind_varnames & {
                    v.name for v in stmt.list_vars()
                }:
                    ind_used = True
                    dprint(f"index of 1D_Var pafor {parfor.id} used in {stmt}")
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

            func_text = (
                ""
                "def _fix_ind_bounds(start, stop):\n"
                "    prefix = bodo.libs.distributed_api.dist_exscan(stop - start, _op)\n"
                "    # rank = bodo.libs.distributed_api.get_rank()\n"
                "    # print(rank, prefix, start, stop)\n"
                "    return start + prefix, stop + prefix\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            f_block = compile_to_numba_ir(
                loc_vars["_fix_ind_bounds"],
                {"bodo": bodo, "_op": np.int32(Reduce_Type.Sum.value)},
                typingctx=self.typingctx,
                targetctx=self.targetctx,
                arg_typs=(types.intp, types.intp),
                typemap=self.typemap,
                calltypes=self.calltypes,
            ).blocks.popitem()[1]
            replace_arg_nodes(f_block, [l_nest.start, l_nest.stop])
            nodes = f_block.body[:-2]
            ret_var = nodes[-1].target
            gen_getitem(l_nest.start, ret_var, 0, self.calltypes, nodes)
            gen_getitem(l_nest.stop, ret_var, 1, self.calltypes, nodes)
            prepend += nodes
            self._1D_Var_parfor_starts[ind_varname] = l_nest.start

            for arr, index, _ in array_accesses:
                if self._index_has_par_index(index, ind_varname):
                    self._1D_Var_array_accesses[arr].append(index)

        init_reduce_nodes, reduce_nodes = self._gen_parfor_reductions(parfor)
        parfor.init_block.body += init_reduce_nodes
        out = prepend + [parfor] + reduce_nodes

        # Tracing here disabled for now (https://bodo.atlassian.net/browse/BE-1213)
        #        # generate performance trace event
        #        event_nodes = self._gen_start_event("Parfor")
        #        ev_var = event_nodes[-1].target
        #        ev_add_attr_nodes = []
        #        ev_add_attr_nodes += self._gen_event_add_attribute(
        #            ev_var, "parfor_ID", str(parfor.id)
        #        )
        #        ev_add_attr_nodes += self._gen_event_add_attribute(
        #            ev_var, "distribution", "1D_Var"
        #        )
        #        finalize_nodes = self._gen_finalize_event(ev_var)
        #        return event_nodes + ev_add_attr_nodes + out + finalize_nodes

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
        """generate distributed reduction calls for parfor reductions"""
        scope = parfor.init_block.scope
        loc = parfor.init_block.loc
        pre = []
        out = []
        for reduce_varname, _reduce_var_info in sorted(parfor.reddict.items()):
            reduce_nodes = _reduce_var_info.reduce_nodes
            reduce_op = guard(
                get_reduce_op, reduce_varname, reduce_nodes, self.func_ir, self.typemap
            )
            reduce_var = reduce_nodes[-1].target
            # TODO: initialize reduction vars (arrays)
            pre += self._gen_init_reduce(reduce_var, reduce_op)
            out += self._gen_reduce(reduce_var, reduce_op, scope, loc)
            # make sure all versions of the reduce variable have the right output
            # SSA changes in Numba 0.53.0rc2 may create extra versions of the reduce
            # variable, see test_df_prod
            for v in reduce_var.versioned_names:
                out.append(ir.Assign(reduce_var, ir.Var(scope, v, loc), loc))

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

    def _run_print(self, print_node, equiv_set):
        """handle Print nodes. 1) avoid duplicate prints for all-REP case.
        2) avoid printing empty slices of distributed arrays.
        """

        # avoid printing empty chunks of distributed arrays/series/dataframes
        if not print_node.vararg and all(
            # If the data is distributed we may have an empty chunk and only want
            # to print the non-empty ranks, except rank 0.
            self._is_1D_or_1D_Var_arr(arg.name)
            or isinstance(guard(get_definition, self.func_ir, arg), ir.Const)
            for arg in print_node.args
        ):
            arg_names = ", ".join(f"arg_{i}" for i in range(len(print_node.args)))
            func_text = (
                f"def impl({arg_names}):\n"
                f"    bodo.libs.distributed_api.print_if_not_empty({arg_names})\n"
            )
            loc_vars = {}
            exec(func_text, {"bodo": bodo}, loc_vars)
            impl = loc_vars["impl"]
            return compile_func_single_block(impl, print_node.args, None, self)

        # avoid replicated prints, print on all PEs only when there is dist arg
        if all(self._is_REP(v.name) and not self._is_rank(v) for v in print_node.args):
            args = print_node.args
            arg_names = ", ".join(f"v{i}" for i in range(len(print_node.args)))
            print_args = arg_names

            # handle vararg like print(*a)
            if print_node.vararg is not None:
                arg_names += "{}vararg".format(", " if args else "")
                print_args += "{}*vararg".format(", " if args else "")
                args.append(print_node.vararg)

            func_text = f"def impl({arg_names}):\n"
            func_text += f"  bodo.libs.distributed_api.single_print({print_args})\n"
            loc_vars = {}
            exec(func_text, {"bodo": bodo}, loc_vars)
            impl = loc_vars["impl"]

            return compile_func_single_block(impl, args, None, self)

        return [print_node]

    def _is_dist_slice(self, var, equiv_set):
        """return True if 'var' is a limited (not full length) distributed slice of a
        distributed array/series/dataframe.
        Limited means the slice doesn't go through the whole length of the array like
        `A[:]` or `A[::3]`. The `is_whole_slice` and `is_slice_equiv_arr ` checks below
        test this case.
        """
        # make sure var is distributed, and is output of getitem of distributed array
        require(self._is_1D_or_1D_Var_arr(var.name))
        var_def = get_definition(self.func_ir, var.name)
        fdef = guard(find_callname, self.func_ir, var_def)
        func_mod = func_name = None
        if fdef is not None:
            func_name, func_mod = fdef

        # dataframe case, check data arrays and index
        if fdef == (
            "init_dataframe",
            "bodo.hiframes.pd_dataframe_ext",
        ):
            arrs_tup = get_definition(self.func_ir, var_def.args[0])
            if isinstance(arrs_tup, ir.Const) and arrs_tup.value == ():
                arrs = [var_def.args[1]]
            else:
                assert is_expr(arrs_tup, "build_tuple"), "invalid init_dataframe"
                arrs = list(arrs_tup.items) + [var_def.args[1]]
            return all(self._is_dist_slice(v, equiv_set) for v in arrs)

        # Series case, check data array and index
        if fdef == (
            "init_series",
            "bodo.hiframes.pd_series_ext",
        ):
            return self._is_dist_slice(
                var_def.args[0], equiv_set
            ) and self._is_dist_slice(var_def.args[1], equiv_set)

        # Index case, check data array
        if func_mod == "bodo.hiframes.pd_index_ext" and func_name in (
            "init_numeric_index",
            "init_binary_str_index",
            "init_categorical_index",
            "init_datetime_index",
            "init_timedelta_index",
            "init_period_index",
            "init_interval_index",
            "get_index_data",
        ):
            return self._is_dist_slice(var_def.args[0], equiv_set)

        if fdef == ("table_filter", "bodo.hiframes.table"):
            require(self._is_1D_or_1D_Var_arr(var_def.args[0].name))
            index_var = var_def.args[1]
        else:
            # array case
            require(
                isinstance(var_def, ir.Expr)
                and var_def.op in ("getitem", "static_getitem")
            )
            require(self._is_1D_or_1D_Var_arr(var_def.value.name))
            index_var = get_getsetitem_index_var(var_def, self.typemap, [])

        # make sure index is a limited slice
        require(self.typemap[index_var.name] in (types.slice2_type, types.slice3_type))
        require(
            not guard(
                is_whole_slice,
                self.typemap,
                self.func_ir,
                index_var,
                accept_stride=True,
            )
        )
        require(
            not guard(
                is_slice_equiv_arr,
                var,
                index_var,
                self.func_ir,
                equiv_set,
                accept_stride=True,
            )
        )
        return True

    def _is_rank(self, v):
        """return True if 'v' is output of bodo.get_rank()"""
        var_def = guard(get_definition, self.func_ir, v.name)
        return guard(find_callname, self.func_ir, var_def) == ("get_rank", "bodo")

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
                eval("lambda A: len(A)"),
                {},
                typingctx=self.typingctx,
                targetctx=self.targetctx,
                arg_typs=(self.typemap[arr.name],),
                typemap=self.typemap,
                calltypes=self.calltypes,
            ).blocks.popitem()[1]
            replace_arg_nodes(f_block, [arr])
            nodes = f_block.body[:-3]  # remove none return
            count_var = nodes[-1].target
            return nodes, start_var, count_var

        size_var = self._get_dist_var_len(arr, nodes, equiv_set, avail_vars)
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
            size_var = self._get_dist_var_len(arr, nodes, equiv_set, avail_vars)
            start_var = self._get_1D_start(size_var, avail_vars, nodes)
        else:
            assert self._is_1D_Var_arr(arr.name)
            nodes = compile_func_single_block(
                eval(
                    "lambda arr: bodo.libs.distributed_api.dist_exscan(len(arr), _op)"
                ),
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
        func_text = (
            ""
            "def f(A, dim, op):\n"
            "    c = A.shape[dim]\n"
            "    res = bodo.libs.distributed_api.dist_reduce(c, op)\n"
        )

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        f_block = compile_to_numba_ir(
            loc_vars["f"],
            {"bodo": bodo},
            typingctx=self.typingctx,
            targetctx=self.targetctx,
            arg_typs=(self.typemap[var.name], types.int64, types.int32),
            typemap=self.typemap,
            calltypes=self.calltypes,
        ).blocks.popitem()[1]
        replace_arg_nodes(
            f_block,
            [var, ir.Const(dim, var.loc), ir.Const(Reduce_Type.Sum.value, var.loc)],
        )
        nodes += f_block.body[:-3]  # remove none return
        return nodes[-1].target

    def _get_dist_var_len(self, var, nodes, equiv_set, avail_vars):
        """
        Get global length of distributed data structure (Array, Series,
        DataFrame) if available. Otherwise, generate reduction code to get
        global length.
        """
        shape = equiv_set.get_shape(var)
        if (
            isinstance(shape, (list, tuple))
            and len(shape) > 0
            and shape[0].name in avail_vars
        ):
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
        func_text = (
            ""
            "def f(A, op):\n"
            "    c = len(A)\n"
            "    res = bodo.libs.distributed_api.dist_reduce(c, op)\n"
        )
        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        f_block = compile_to_numba_ir(
            loc_vars["f"],
            {"bodo": bodo},
            typingctx=self.typingctx,
            targetctx=self.targetctx,
            arg_typs=(self.typemap[arr.name], types.int32),
            typemap=self.typemap,
            calltypes=self.calltypes,
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
            eval("lambda n, rank, n_pes: rank * (n // n_pes) + min(rank, n % n_pes)"),
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

        func_text = (
            ""
            "def impl(n, rank, n_pes):\n"
            "    res = n % n_pes\n"
            "    # The formula we would like is if (rank < res): blk_size +=1 but this does not compile\n"
            "    blk_size = n // n_pes + min(rank + 1, res) - min(rank, res)\n"
            "    return blk_size\n"
        )

        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        nodes += compile_func_single_block(
            loc_vars["impl"], (size_var, self.rank_var, self.n_pes_var), None, self
        )
        count_var = nodes[-1].target
        # rename for readability
        count_var.name = mk_unique_var("count_var")
        self.typemap[count_var.name] = types.int64
        return count_var

    def _get_1D_end(self, size_var, nodes):
        """get end index of size_var in 1D_Block distribution"""
        nodes += compile_func_single_block(
            eval(
                "lambda n, rank, n_pes: (rank + 1) * (n // n_pes) + min(rank + 1, n % n_pes)"
            ),
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
            eval("lambda ind, start: ind - start"),
            {},
            typingctx=self.typingctx,
            targetctx=self.targetctx,
            arg_typs=(types.intp, types.intp),
            typemap=self.typemap,
            calltypes=self.calltypes,
        )
        block = f_ir.blocks.popitem()[1]
        replace_arg_nodes(block, [ind_var, start_var])
        return block.body[:-2]

    def _get_ind_sub_slice(self, slice_var, offset_var):
        if isinstance(slice_var, slice):
            f_text = f"""def f(offset):
                return slice({slice_var.start} - offset, {slice_var.stop} - offset)
            """
            loc = {}
            exec(f_text, {}, loc)
            f = loc["f"]
            args = [offset_var]
            arg_typs = (types.intp,)
        else:
            func_text = (
                ""
                "def f(old_slice, offset):\n"
                "    return slice(old_slice.start - offset, old_slice.stop - offset)\n"
            )

            loc_vars = {}
            exec(func_text, globals(), loc_vars)
            f = loc_vars["f"]
            args = [slice_var, offset_var]
            slice_type = self.typemap[slice_var.name]
            arg_typs = (slice_type, types.intp)
        _globals = self.func_ir.func_id.func.__globals__
        f_ir = compile_to_numba_ir(
            f,
            _globals,
            typingctx=self.typingctx,
            targetctx=self.targetctx,
            arg_typs=arg_typs,
            typemap=self.typemap,
            calltypes=self.calltypes,
        )
        _, block = f_ir.blocks.popitem()
        replace_arg_nodes(block, args)
        return block.body[:-2]  # ignore return nodes

    def _file_open_set_parallel(self, file_varname):
        """Finds file open call (h5py.File) for file_varname and sets the parallel flag."""
        from bodo.io.h5_api import h5file_type, h5group_type

        # TODO: find and handle corner cases
        var = file_varname
        while True:
            var_def = get_definition(self.func_ir, var)
            require(isinstance(var_def, ir.Expr))
            if var_def.op == "call":
                fdef = find_callname(self.func_ir, var_def, self.typemap)
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

    def _gen_reduce(self, reduce_var, reduce_op, scope, loc):
        """generate distributed reduction code for after parfor's local execution"""
        # concat reduction variables don't need aggregation since output is distributed
        # see test_concat_reduction
        if reduce_op in (Reduce_Type.Concat, Reduce_Type.No_Op):
            return []

        op_var = ir.Var(scope, mk_unique_var("$reduce_op"), loc)
        self.typemap[op_var.name] = types.int32
        op_assign = ir.Assign(ir.Const(reduce_op.value, loc), op_var, loc)
        # keep local versions of reduce variables to enable converting variable size
        # string allocations to local chunk size
        local_reduce_var = ir.Var(scope, mk_unique_var(reduce_var.name), loc)
        self.typemap[local_reduce_var.name] = self.typemap[reduce_var.name]
        local_assign = ir.Assign(reduce_var, local_reduce_var, loc)
        self._local_reduce_vars[reduce_var.name] = local_reduce_var
        dist_reduce_nodes = [local_assign, op_assign] + compile_func_single_block(
            eval("lambda val, op: bodo.libs.distributed_api.dist_reduce(val, op)"),
            [reduce_var, op_var],
            reduce_var,
            self,
        )

        return dist_reduce_nodes

    def _gen_init_reduce(self, reduce_var, reduce_op):
        """generate code to initialize reduction variables on non-root
        processors.
        Avoids extra code generation if the user code-specified init value is the same
        as the neutral value of the reduction (common case).
        """
        # TODO: support initialization for concat reductions
        if reduce_op in (Reduce_Type.Concat, Reduce_Type.No_Op):
            return []

        extra_globals = {}

        red_var_typ = self.typemap[reduce_var.name]
        el_typ = red_var_typ
        if is_np_array_typ(self.typemap[reduce_var.name]):
            el_typ = red_var_typ.dtype
        init_val = None
        user_init_val = self._get_reduce_user_init(reduce_var)
        pre_init_val = ""

        if reduce_op in [Reduce_Type.Sum, Reduce_Type.Bit_Or]:
            init_val = str(el_typ(0))
            if user_init_val == el_typ(0):
                return []
        if reduce_op == Reduce_Type.Prod:
            init_val = str(el_typ(1))
            if user_init_val == el_typ(1):
                return []
        if reduce_op == Reduce_Type.Min:
            if el_typ == types.bool_:
                init_val = "True"
                if user_init_val == True:
                    return []
            elif isinstance(el_typ, bodo.Decimal128Type):
                extra_globals["_str_to_decimal_scalar"] = (
                    bodo.libs.decimal_arr_ext._str_to_decimal_scalar
                )
                extra_globals["prec"] = el_typ.precision
                extra_globals["scale"] = el_typ.scale
                dec_str_setup = "'9' * (prec - scale) + '.' + '9' * scale"
                pre_init_val = (
                    f"val, _ = _str_to_decimal_scalar({dec_str_setup}, prec, scale)"
                )
                init_val = "val"
            elif el_typ == bodo.datetime_date_type:
                init_val = "bodo.hiframes.series_kernels._get_date_max_value()"
            elif isinstance(el_typ, TimeType):
                init_val = "bodo.hiframes.series_kernels._get_time_max_value()"
            elif isinstance(el_typ, bodo.TimestampTZType):
                init_val = "bodo.hiframes.series_kernels._get_timestamptz_max_value()"
            else:
                init_val = f"numba.cpython.builtins.get_type_max_value(np.ones(1,dtype=np.{el_typ}).dtype)"
        if reduce_op == Reduce_Type.Max:
            if el_typ == types.bool_:
                init_val = "False"
                if user_init_val == False:
                    return []
            elif isinstance(el_typ, bodo.Decimal128Type):
                extra_globals["_str_to_decimal_scalar"] = (
                    bodo.libs.decimal_arr_ext._str_to_decimal_scalar
                )
                extra_globals["prec"] = el_typ.precision
                extra_globals["scale"] = el_typ.scale
                dec_str_setup = "'-' + '9' * (prec - scale) + '.' + '9' * scale"
                pre_init_val = (
                    f"val, _ = _str_to_decimal_scalar({dec_str_setup}, prec, scale)"
                )
                init_val = "val"
            elif el_typ == bodo.datetime_date_type:
                init_val = "bodo.hiframes.series_kernels._get_date_min_value()"
            elif isinstance(el_typ, TimeType):
                init_val = "bodo.hiframes.series_kernels._get_time_min_value()"
            elif isinstance(el_typ, bodo.TimestampTZType):
                init_val = "bodo.hiframes.series_kernels._get_timestamptz_min_value()"
            else:
                init_val = f"numba.cpython.builtins.get_type_min_value(np.ones(1,dtype=np.{el_typ}).dtype)"
        if reduce_op in [Reduce_Type.Argmin, Reduce_Type.Argmax]:
            # don't generate initialization for argmin/argmax since they are not
            # initialized by user and correct initialization is already there
            return []

        assert init_val is not None, "Invalid distributed reduction"

        if is_np_array_typ(self.typemap[reduce_var.name]):
            pre_init_val = f"v = np.full_like(s, {init_val}, s.dtype)"
            init_val = "v"

        f_text = f"def f(s):\n  {pre_init_val}\n  return bodo.libs.distributed_api._root_rank_select(s, {init_val})"
        loc_vars = {}
        exec(
            f_text, {"bodo": bodo, "numba": numba, "np": np, **extra_globals}, loc_vars
        )
        f = loc_vars["f"]

        return compile_func_single_block(
            f,
            (reduce_var,),
            reduce_var,
            self,
        )

    def _get_reduce_user_init(self, reduce_var):
        """get the initialization value provided by user code for the reduction
        variable.
        This assumes the reduce variable has two definitions, one being initialization
        and the other update in the parfor loop.
        """
        reduce_var_defs = self.func_ir._definitions[reduce_var.name]
        if len(reduce_var_defs) != 2:
            return None

        for var_def in reduce_var_defs:
            if isinstance(var_def, ir.Var):
                var_def = guard(get_definition, self.func_ir, var_def)
            if isinstance(var_def, (ir.Const, ir.FreeVar, ir.Global)):
                return var_def.value

        return None  # init value not found

    def _gen_init_code(self, blocks):
        """generate get_rank() and get_size() calls and store the variables
        to avoid repeated generation.
        """
        # get rank variable
        nodes = compile_func_single_block(
            eval("lambda: _get_rank()"),
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
            eval("lambda: _get_size()"),
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

    def _set_ith_arg_to_unliteral(self, rhs: ir.Expr, i: int) -> None:
        """Set the ith argument of call expr 'rhs' to a nonliteral version.
        This assumes the ith input is a literal. This is used for Bodo function replacements
        that potentially replace an original constant in the IR with a variable.

        For example this would be used if we had to make the following replacment:

            f(0) -> f(int_var)

        Args:
            rhs (ir.Expr): Call expression with at least i + 1 arguments and the ith argument is
            a literal.
        """
        call_type = self.calltypes[rhs]
        # In some cases the call type is already not a literal when the argument
        # is a literal. As a result we only change the call if its a literal.
        if isinstance(call_type.args[i], types.Literal):
            self.calltypes.pop(rhs)
            self.calltypes[rhs] = self.typemap[rhs.func.name].get_call_type(
                self.typingctx,
                call_type.args[:i]
                + (types.unliteral(call_type.args[i]),)
                + call_type.args[i + 1 :],
                {},
            )

    def _set_second_last_arg_to_true(self, rhs):
        """set second-to-last argument of call expr 'rhs' to True, assuming that it is an
        Omitted arg with value of False.
        This is usually used for Bodo overloads that have an extra flag as second-to-last
        argument to enable parallelism.
        """
        call_type = self.calltypes.pop(rhs)
        assert call_type.args[-2] == types.Omitted(False)
        self.calltypes[rhs] = self.typemap[rhs.func.name].get_call_type(
            self.typingctx,
            call_type.args[:-2] + (types.Omitted(True), call_type.args[-1]),
            {},
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
        """Returns True if input statement is a form of array access, e.g. equivalent to
        A[i].
        This allows the compiler to handle 1D_Var parallelism properly without expensive
        exscan calls.
        NOTE: all internal array access nodes/functions have to be handled here.

        Example without exscan:
        for i in range(0, local_len):
            isna(A, i)

        Example with exscan:
        prefix = exscan(local_len)
        for i in range(prefix, prefix + local_len):
            isna(A, i)

        Args:
            stmt (ir.Stmt): input statement

        Returns:
            bool: true if input is an array access statement
        """
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
                ("get_str_arr_str_length", "bodo.libs.str_arr_ext"),
                ("inplace_eq", "bodo.libs.str_arr_ext"),
                ("get_str_arr_item_copy", "bodo.libs.str_arr_ext"),
                ("copy_array_element", "bodo.libs.array_kernels"),
                ("str_arr_setitem_int_to_str", "bodo.libs.str_arr_ext"),
                ("str_arr_setitem_NA_str", "bodo.libs.str_arr_ext"),
                ("str_arr_set_not_na", "bodo.libs.str_arr_ext"),
                ("get_split_view_index", "bodo.hiframes.split_impl"),
                ("get_bit_bitmap_arr", "bodo.libs.int_arr_ext"),
                ("set_bit_to_arr", "bodo.libs.int_arr_ext"),
                ("scalar_optional_getitem", "bodo.utils.indexing"),
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
            f_text += f"  val{i} = tup_var[{i}]\n"
        loc_vars = {}
        exec(f_text, {}, loc_vars)
        f = loc_vars["f"]
        f_block = compile_to_numba_ir(
            f,
            {},
            typingctx=self.typingctx,
            targetctx=self.targetctx,
            arg_typs=(self.typemap[tup_var.name],),
            typemap=self.typemap,
            calltypes=self.calltypes,
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

    def _try_meta_head(self, io_node):
        """
        Check if reading metadata and/or head rows from a data source is enough for the
        program. If so, returns the read size and the variable for setting the total
        size.
        Also, replaces array shapes in the IR with the total size variable.
        Raises GuardException if this optimization is not possible.

        Only tested and used when reading from Parquet Files or an Iceberg DB
        """
        arr_varnames = {v.name for v in io_node.out_vars}
        # arr.shape nodes to transform
        shape_nodes = []
        read_size = None
        scope = None

        # make sure arrays are only used in arr.shape or arr[:11] cases
        # traverse in topo order to see get_table_data() def output before use
        for label in find_topo_order(self.func_ir.blocks):
            block = self.func_ir.blocks[label]
            for stmt in block.body:
                # pq read node
                if stmt is io_node:
                    scope = block.scope
                    continue
                if is_assign(stmt):
                    rhs = stmt.value
                    # arr.shape match
                    if (
                        is_expr(rhs, "getattr")
                        and rhs.attr == "shape"
                        and rhs.value.name in arr_varnames
                    ):
                        shape_nodes.append(stmt)
                        continue
                    if (
                        is_call(rhs)
                        and guard(find_callname, self.func_ir, rhs)
                        == (
                            "table_subset",
                            "bodo.hiframes.table",
                        )
                        and rhs.args[0].name in arr_varnames
                    ):
                        # If we take a subset of a table treat it as a
                        # used array/table.
                        arr_varnames.add(stmt.target.name)
                        continue

                    # arr[:11] match for both array and table format
                    if (
                        is_call(rhs)
                        and guard(find_callname, self.func_ir, rhs)
                        == (
                            "table_filter",
                            "bodo.hiframes.table",
                        )
                        and (rhs.args[0].name in arr_varnames)
                    ) or (is_expr(rhs, "getitem") and rhs.value.name in arr_varnames):
                        if is_call(rhs):
                            # table_format
                            index_def = get_definition(self.func_ir, rhs.args[1])
                        else:
                            # array_format
                            # TODO(ehsan): Numba may produce static_getitem?
                            index_def = get_definition(self.func_ir, rhs.index)

                        require(
                            find_callname(self.func_ir, index_def)
                            == ("slice", "builtins")
                        )
                        require(len(index_def.args) == 2)
                        require(
                            find_const(self.func_ir, index_def.args[0]) in (None, 0)
                        )
                        # corner case: if there are multiple filters on the same table,
                        # make sure they all read the same amount of data
                        # TODO[BE-3580]: support non-constant sizes
                        slice_size = get_const_value_inner(
                            self.func_ir, index_def.args[1], typemap=self.typemap
                        )
                        if read_size is None:
                            read_size = slice_size
                        else:
                            require(slice_size == read_size)
                        continue
                    if (
                        is_call(rhs)
                        and guard(find_callname, self.func_ir, rhs)
                        == (
                            "get_table_data",
                            "bodo.hiframes.table",
                        )
                        and rhs.args[0].name in arr_varnames
                    ):
                        arr_varnames.add(stmt.target.name)
                        continue
                    if (
                        is_call(rhs)
                        and guard(find_callname, self.func_ir, rhs)
                        == (
                            "set_table_data_null",
                            "bodo.hiframes.table",
                        )
                        and rhs.args[0].name in arr_varnames
                    ):
                        # If we are just replacing a column with null then we can
                        # still safely perform filter pushdown.
                        arr_varnames.add(stmt.target.name)
                        continue
                    if isinstance(rhs, ir.Var) and rhs.name in arr_varnames:
                        # If we have a simple alias we just need to track the new lhs
                        arr_varnames.add(stmt.target.name)
                        continue

                # other nodes, array variables shouldn't be used
                require(not (arr_varnames & {v.name for v in stmt.list_vars()}))

        # optimization is possible, replace the total size variable in shape nodes
        require(scope is not None)
        # If read size is None we don't have limit pushdown
        require(read_size is not None)
        total_len_var = ir.Var(scope, mk_unique_var("total_df_len"), io_node.loc)
        self.typemap[total_len_var.name] = types.int64
        for stmt in shape_nodes:
            stmt.value = ir.Expr.build_tuple([total_len_var], stmt.loc)

        log_limit_pushdown(io_node, read_size)

        return read_size, total_len_var

    def _fix_set_node_sig(self, inst):
        """update call signature of setitem/static_setitem/setattr in calltypes
        since distributed analysis can change distribution of dataframes/series types
        example: test_series_loc_setitem_bool
        """
        # should not be possible but just in case
        if inst not in self.calltypes:  # pragma: no cover
            return

        sig = self.calltypes.pop(inst)

        # should not be possible but just in case
        if sig is None:  # pragma: no cover
            self.calltypes[inst] = None
            return

        sig = sig.replace(args=(self.typemap[inst.target.name],) + sig.args[1:])
        self.calltypes[inst] = sig

    def _get_arr_ndim(self, arrname):
        if is_str_arr_type(self.typemap[arrname]):
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

    def _gen_start_event(self, event_name):
        """generate Event() call nodes with 'event_name' input"""
        return compile_func_single_block(
            eval(f"lambda: Event('{event_name}')"),
            [],
            None,
            self,
            extra_globals={"Event": bodo.utils.tracing.Event},
        )

    def _gen_event_add_attribute(self, ev_var, name, value):
        """generate ev.add_attribute() call nodes with 'name' and 'value' inputs"""
        if isinstance(value, str):
            value_str = f"'{value}'"
        else:
            value_str = f"{value}"
        return compile_func_single_block(
            eval(f"lambda ev: ev.add_attribute('{name}', {value_str})"),
            [ev_var],
            None,
            self,
        )

    def _gen_finalize_event(self, ev_var):
        """generate event.finalize() call nodes"""
        return compile_func_single_block(
            eval("lambda ev: ev.finalize()"),
            [ev_var],
            None,
            self,
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
def lower_parfor_sequential(typingctx, func_ir, typemap, calltypes, metadata):
    ir_utils._the_max_label.update(ir_utils.find_max_label(func_ir.blocks))
    parfor_found = False
    new_blocks = {}
    scope = next(iter(func_ir.blocks.values())).scope
    for block_label, block in func_ir.blocks.items():
        block_label, parfor_found = _lower_parfor_sequential_block(
            block_label,
            block,
            new_blocks,
            typemap,
            calltypes,
            parfor_found,
            scope=scope,
        )
        # old block stays either way
        new_blocks[block_label] = block
    func_ir.blocks = new_blocks
    # rename only if parfor found and replaced (avoid test_flow_control error)
    if parfor_found:
        func_ir.blocks = rename_labels(func_ir.blocks)
    dprint_func_ir(func_ir, "after parfor sequential lowering")
    simplify(func_ir, typemap, calltypes, metadata)
    dprint_func_ir(func_ir, "after parfor sequential simplify")
    # changed from Numba code: comment out id.Del generation that causes errors in 0.49
    # # add dels since simplify removes dels
    # post_proc = postproc.PostProcessor(func_ir)
    # post_proc.run(True)


if bodo.numba_compat._check_numba_change:
    lines = inspect.getsource(numba.parfors.parfor.lower_parfor_sequential)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "c548919f8821df2f137b18cb16791ab5761c1f6ff678864893bb758abd7dc2b1"
    ):  # pragma: no cover
        warnings.warn("numba.parfors.parfor.lower_parfor_sequential has changed")
