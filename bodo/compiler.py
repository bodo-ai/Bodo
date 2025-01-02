"""
Defines Bodo's compiler pipeline.
"""

import os
import warnings
from collections import namedtuple

import numba
from numba.core import ir, ir_utils, types
from numba.core.compiler import DefaultPassBuilder
from numba.core.compiler_machinery import (
    AnalysisPass,
    FunctionPass,
    register_pass,
)
from numba.core.errors import (
    NumbaExperimentalFeatureWarning,
    NumbaPendingDeprecationWarning,
)
from numba.core.inline_closurecall import inline_closure_call
from numba.core.ir_utils import (
    build_definitions,
    find_callname,
    get_definition,
    guard,
)
from numba.core.registry import CPUDispatcher
from numba.core.typed_passes import (
    DumpParforDiagnostics,
    InlineOverloads,
    IRLegalization,
    NopythonTypeInference,
    ParforPreLoweringPass,
    PreParforPass,
)
from numba.core.untyped_passes import (
    MakeFunctionToJitFunction,
    ReconstructSSA,
    WithLifting,
)

import bodo
import bodo.decorators
import bodo.hiframes.dataframe_indexing  # noqa # side effect: initialize Numba extensions
import bodo.hiframes.datetime_datetime_ext  # noqa # side effect: initialize Numba extensions
import bodo.hiframes.datetime_timedelta_ext  # noqa # side effect: initialize Numba extensions
import bodo.io
import bodo.libs
import bodo.libs.array_kernels  # noqa # side effect: install Numba functions
import bodo.libs.int_arr_ext  # noqa # side effect
import bodo.libs.re_ext  # noqa # side effect: initialize Numba extensions
import bodo.libs.spark_extra
import bodo.transforms
import bodo.transforms.series_pass
import bodo.transforms.type_inference
import bodo.transforms.type_inference.typeinfer  # noqa # side effect: initialize Numba extensions
import bodo.transforms.untyped_pass
import bodo.utils
import bodo.utils.table_utils  # noqa # side effect
import bodo.utils.typing
from bodo.transforms.series_pass import SeriesPass
from bodo.transforms.table_column_del_pass import TableColumnDelPass
from bodo.transforms.typing_pass import BodoTypeInference
from bodo.transforms.untyped_pass import UntypedPass
from bodo.utils.utils import is_assign, is_call_assign, is_expr

# avoid Numba warning for UDFs: "First-class function type feature is experimental"
warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
# avoid Numba warning when there is a list argument to JIT function
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

# Numba performance warning is disabled by setting environment variable
# NUMBA_DISABLE_PERFORMANCE_WARNINGS = 1 in __init__.py

# global flag for whether all Bodo functions should be inlined
inline_all_calls = False


class BodoCompiler(numba.core.compiler.CompilerBase):
    """Bodo compiler pipeline which adds the following passes to Numba's pipeline:
    InlinePass, BodoUntypedPass, BodoTypeInference, BodoSeriesPass,
    LowerParforSeq, BodoDumpDistDiagnosticsPass.
    See class docstrings for more info.
    """

    avoid_copy_propagation = False

    def define_pipelines(self):
        return self._create_bodo_pipeline(
            distributed=True, inline_calls_pass=inline_all_calls
        )

    def _create_bodo_pipeline(
        self, distributed=True, inline_calls_pass=False, udf_pipeline=False
    ):
        """create compiler pipeline for Bodo using Numba's nopython pipeline"""
        name = "bodo" if distributed else "bodo_seq"
        name = name + "_inline" if inline_calls_pass else name
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state, name)

        # inline other jit functions right after IR is available
        # NOTE: calling after WithLifting since With blocks should be handled before
        # simplify_CFG() is called (block number is used in EnterWith nodes)
        if inline_calls_pass:
            pm.add_pass_after(InlinePass, WithLifting)
        if udf_pipeline:
            pm.add_pass_after(ConvertCallsUDFPass, WithLifting)
        # run untyped pass right before SSA construction and type inference
        # NOTE: SSA includes phi nodes (which have block numbers) that we don't handle.
        # therefore, uptyped pass cannot use SSA since it changes CFG
        add_pass_before(pm, BodoUntypedPass, ReconstructSSA)
        # replace Numba's type inference pass with Bodo's version, which incorporates
        # constant inference using partial type inference
        replace_pass(pm, BodoTypeInference, NopythonTypeInference)
        # remove make_function conversion pass since it is handled in typing pass now
        remove_pass(pm, MakeFunctionToJitFunction)

        # Series pass should be before pre_parfor since
        # S.call to np.call transformation is invalid for
        # Series (e.g. S.var is not the same as np.var(S))
        add_pass_before(pm, BodoSeriesPass, PreParforPass)
        if distributed:
            pm.add_pass_after(BodoDistributedPass, ParforPreLoweringPass)
        else:
            pm.add_pass_after(LowerParforSeq, ParforPreLoweringPass)
            pm.add_pass_after(LowerBodoIRExtSeq, LowerParforSeq)

        # Decref right before dels are inserted so the IR won't change anymore.
        add_pass_before(pm, BodoTableColumnDelPass, IRLegalization)
        pm.add_pass_after(BodoDumpDistDiagnosticsPass, DumpParforDiagnostics)
        pm.finalize()
        return [pm]


# TODO: remove this helper function when available in Numba
def add_pass_before(pm, pass_cls, location):
    """
    Add a pass `pass_cls` to the PassManager's compilation pipeline right before
    the pass `location`.
    """
    # same as add_pass_after, except first argument to "insert"
    assert pm.passes
    pm._validate_pass(pass_cls)
    pm._validate_pass(location)
    for idx, (x, _) in enumerate(pm.passes):
        if x == location:
            break
    else:  # pragma: no cover
        raise bodo.utils.typing.BodoError(f"Could not find pass {location}")
    pm.passes.insert(idx, (pass_cls, str(pass_cls)))
    # if a pass has been added, it's not finalized
    pm._finalized = False


def replace_pass(pm, pass_cls, location):
    """
    Replace pass `location` in PassManager's compilation pipeline with the pass
    `pass_cls`.
    """
    assert pm.passes
    pm._validate_pass(pass_cls)
    pm._validate_pass(location)
    for idx, (x, _) in enumerate(pm.passes):
        if x == location:
            break
    else:  # pragma: no cover
        raise bodo.utils.typing.BodoError(f"Could not find pass {location}")
    pm.passes[idx] = (pass_cls, str(pass_cls))
    # if a pass has been added, it's not finalized
    pm._finalized = False


def remove_pass(pm, location):
    """
    Remove pass `location` in PassManager's compilation pipeline
    """
    assert pm.passes
    pm._validate_pass(location)
    for idx, (x, _) in enumerate(pm.passes):
        if x == location:
            break
    else:  # pragma: no cover
        raise bodo.utils.typing.BodoError(f"Could not find pass {location}")
    pm.passes.pop(idx)
    # if a pass has been added, it's not finalized
    pm._finalized = False


# TODO: use Numba's new inline feature
@register_pass(mutates_CFG=True, analysis_only=False)
class InlinePass(FunctionPass):
    """inline other jit functions, mainly to enable automatic parallelism"""

    _name = "inline_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Inline function calls (to enable distributed pass analysis)
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        inline_calls(state.func_ir, state.locals)
        # sometimes type inference fails after inlining since blocks are inserted
        # at the end and there are agg constraints (categorical_split case)
        # CFG simplification fixes this case
        state.func_ir.blocks = ir_utils.simplify_CFG(state.func_ir.blocks)
        return True


def _convert_bodo_dispatcher_to_udf(rhs, func_ir):
    """Update Bodo dispatcher calls to use the UDF pipeline (e.g. set distribued=False)"""

    # find the actual function value, could be in current global names (e.g. "myfunc()")
    # or called from another module (e.g. "mymod.myfunc()")
    func_def = guard(get_definition, func_ir, rhs.func)
    if isinstance(func_def, (ir.Global, ir.FreeVar, ir.Const)):
        func_val = func_def.value
    else:
        # match "mymod.myfunc()" call pattern and get the global function value
        fdef = guard(find_callname, func_ir, rhs)
        if not (fdef and isinstance(fdef[0], str) and isinstance(fdef[1], str)):
            return

        func_name, func_mod = fdef
        try:
            import importlib

            mod = importlib.import_module(func_mod)
            func_val = getattr(mod, func_name)
        except Exception:
            return

    # replace regular bodo compiler pipeline with bodo udf pipeline
    if (
        isinstance(func_val, CPUDispatcher)
        and issubclass(func_val._compiler.pipeline_class, BodoCompiler)
        and func_val._compiler.pipeline_class != BodoCompilerUDF
    ):
        func_val._compiler.pipeline_class = BodoCompilerUDF
        # the function may have been compiled in typing of outer functions before
        # getting here so need to recompile to make it sequential.
        # See [BE-2225].
        func_val.recompile()


@register_pass(mutates_CFG=True, analysis_only=False)
class ConvertCallsUDFPass(FunctionPass):
    """Make sure all JUT functions called inside UDFs use the UDF pipeline to avoid
    distributed code generation (which would lead to hangs).
    """

    _name = "inline_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Convert Bodo calls to use the UDF pipeline
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        for block in state.func_ir.blocks.values():
            for inst in block.body:
                if is_call_assign(inst):
                    _convert_bodo_dispatcher_to_udf(inst.value, state.func_ir)

        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class BodoUntypedPass(FunctionPass):
    """
    Transformations before typing to enable type inference.
    This pass transforms the IR to remove operations that cannot be handled in Numba's
    type inference due to complexity such as pd.read_csv().
    """

    _name = "bodo_untyped_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Fix IR before typing to handle untypeable cases
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        untyped_pass = UntypedPass(
            state.func_ir,
            state.typingctx,
            state.args,
            state.locals,
            state.metadata,
            state.flags,
            isinstance(state.pipeline, BodoCompilerSeq),
        )
        untyped_pass.run()
        return True


def _update_definitions(func_ir, node_list):
    """update variable definition lists in IR for new list of statements"""
    loc = ir.Loc("", 0)
    dumm_block = ir.Block(ir.Scope(None, loc), loc)
    dumm_block.body = node_list
    build_definitions({0: dumm_block}, func_ir._definitions)


_series_inline_attrs = {"values", "shape", "size", "empty", "name", "index", "dtype"}
# Series methods that are not inlined currently (may be able to inline later)
_series_no_inline_methods = {
    "to_list",
    "tolist",
    "rolling",
    "to_csv",
    "count",
    "fillna",
    "to_dict",
    "map",
    "apply",
    "pipe",
    "combine",
    "bfill",
    "ffill",
    "pad",
    "backfill",
    "mask",
    "where",
}
# Series methods that are just aliases of another method
_series_method_alias = {
    "isnull": "isna",
    "product": "prod",
    "kurtosis": "kurt",
    "notnull": "notna",
}
# DataFrame methods that are not inlined currently, but some may be possible to inline
# in the future
_dataframe_no_inline_methods = {
    "apply",
    "itertuples",
    "pipe",
    "to_parquet",
    "to_sql",
    "to_csv",
    "to_json",
    "assign",
    "to_string",
    "query",
    "rolling",
    "mask",
    "where",
}

TypingInfo = namedtuple(
    "TypingInfo", ["typingctx", "targetctx", "typemap", "calltypes", "curr_loc"]
)


def _inline_bodo_getattr(
    stmt, rhs, rhs_type, new_body, func_ir, typingctx, targetctx, typemap, calltypes
):
    """Inline getattr nodes for Bodo types like Series"""
    from bodo.hiframes.pd_dataframe_ext import DataFrameType
    from bodo.hiframes.pd_series_ext import SeriesType
    from bodo.utils.transform import compile_func_single_block

    if isinstance(rhs_type, SeriesType) and rhs.attr in _series_inline_attrs:
        overload_name = "overload_series_" + rhs.attr
        overload_func = getattr(bodo.hiframes.series_impl, overload_name)
    if isinstance(rhs_type, DataFrameType) and rhs.attr in ("index", "columns"):
        overload_name = "overload_dataframe_" + rhs.attr
        overload_func = getattr(bodo.hiframes.dataframe_impl, overload_name)
    else:
        return False

    func_ir._definitions[stmt.target.name].remove(rhs)
    impl = overload_func(rhs_type)
    typing_info = TypingInfo(typingctx, targetctx, typemap, calltypes, stmt.loc)
    nodes = compile_func_single_block(
        impl,
        (rhs.value,),
        stmt.target,
        typing_info,
    )
    _update_definitions(func_ir, nodes)
    new_body += nodes

    return True


def _inline_bodo_call(
    rhs,
    i,
    func_mod,
    func_name,
    pass_info,
    new_body,
    block,
    typingctx,
    targetctx,
    calltypes,
    work_list,
):
    """Inline Bodo calls if possible (e.g. Series method calls)"""
    from bodo.hiframes.pd_dataframe_ext import DataFrameType
    from bodo.hiframes.pd_series_ext import SeriesType
    from bodo.utils.transform import replace_func, update_locs

    func_ir = pass_info.func_ir
    typemap = pass_info.typemap

    # Series method call
    if (
        isinstance(func_mod, ir.Var)
        and isinstance(typemap[func_mod.name], SeriesType)
        and func_name not in _series_no_inline_methods
    ):
        if func_name in _series_method_alias:
            func_name = _series_method_alias[func_name]
        # Series.add/... are implemented by overload generation
        if func_name in bodo.hiframes.series_impl.explicit_binop_funcs or (
            func_name.startswith("r")
            and func_name[1:] in bodo.hiframes.series_impl.explicit_binop_funcs
        ):
            return False
        rhs.args.insert(0, func_mod)
        arg_typs = tuple(typemap[v.name] for v in rhs.args)
        kw_typs = {name: typemap[v.name] for name, v in dict(rhs.kws).items()}
        impl = getattr(bodo.hiframes.series_impl, "overload_series_" + func_name)(
            *arg_typs, **kw_typs
        )
    # DataFrame method call
    elif (
        isinstance(func_mod, ir.Var)
        and isinstance(typemap[func_mod.name], DataFrameType)
        and func_name not in _dataframe_no_inline_methods
    ):
        # dataframe method aliases are the same as Series
        if func_name in _series_method_alias:
            func_name = _series_method_alias[func_name]
        rhs.args.insert(0, func_mod)
        arg_typs = tuple(typemap[v.name] for v in rhs.args)
        kw_typs = {name: typemap[v.name] for name, v in dict(rhs.kws).items()}
        impl = getattr(bodo.hiframes.dataframe_impl, "overload_dataframe_" + func_name)(
            *arg_typs, **kw_typs
        )
    else:
        return False

    rp_func = replace_func(
        pass_info,
        impl,
        rhs.args,
        pysig=numba.core.utils.pysignature(impl),
        kws=dict(rhs.kws),
    )
    block.body = new_body + block.body[i:]
    callee_blocks, _ = inline_closure_call(
        func_ir,
        rp_func.glbls,
        block,
        len(new_body),
        rp_func.func,
        typingctx=typingctx,
        targetctx=targetctx,
        arg_typs=rp_func.arg_types,
        typemap=typemap,
        calltypes=calltypes,
        work_list=work_list,
    )
    # update Loc objects
    for c_block in callee_blocks.values():
        c_block.loc = rhs.loc
        update_locs(c_block.body, rhs.loc)

    return True


def bodo_overload_inline_pass(func_ir, typingctx, targetctx, typemap, calltypes):
    """inline Bodo overloads to make compilation time faster than with Numba's inliner.
    Single block functions for example can be inlined faster here.

    Adding all Bodo overloads here is not necessary for correctness, but adding the
    common functions is important for faster compilation time.
    """

    PassInfo = namedtuple("PassInfo", ["func_ir", "typemap"])
    pass_info = PassInfo(func_ir, typemap)

    blocks = func_ir.blocks
    work_list = [(l, blocks[l]) for l in reversed(blocks.keys())]
    while work_list:
        label, block = work_list.pop()
        new_body = []
        replaced = False

        for i, stmt in enumerate(block.body):
            # inline getattr if possible
            if is_assign(stmt) and is_expr(stmt.value, "getattr"):
                rhs = stmt.value
                rhs_type = typemap[rhs.value.name]
                if _inline_bodo_getattr(
                    stmt,
                    rhs,
                    rhs_type,
                    new_body,
                    func_ir,
                    typingctx,
                    targetctx,
                    typemap,
                    calltypes,
                ):
                    continue
            # inline call if possible
            if is_call_assign(stmt):
                rhs = stmt.value
                fdef = guard(find_callname, func_ir, rhs, typemap)
                if fdef is None:
                    new_body.append(stmt)
                    continue
                func_name, func_mod = fdef
                if _inline_bodo_call(
                    rhs,
                    i,
                    func_mod,
                    func_name,
                    pass_info,
                    new_body,
                    block,
                    typingctx,
                    targetctx,
                    calltypes,
                    work_list,
                ):
                    replaced = True
                    break
            new_body.append(stmt)

        if not replaced:
            blocks[label].body = new_body

    func_ir.blocks = ir_utils.simplify_CFG(func_ir.blocks)


@register_pass(mutates_CFG=True, analysis_only=False)
class BodoDistributedPass(FunctionPass):
    """
    This pass analyzes the IR to decide parallelism of arrays and parfors for
    distributed transformation, then parallelizes the IR for distributed execution and
    inserts MPI calls.
    Specialized IR nodes are also transformed to regular IR here since all analysis and
    transformations are done.
    """

    _name = "bodo_distributed_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        parallelize for distributed-memory
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        from bodo.transforms.distributed_pass import DistributedPass

        dist_pass = DistributedPass(
            state.func_ir,
            state.typingctx,
            state.targetctx,
            state.typemap,
            state.calltypes,
            state.return_type,
            state.metadata,
            state.flags,
        )
        # update in distributed analysis if distribution hint of return type changed
        # (results in lowering error otherwise)
        state.return_type = dist_pass.run()
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class BodoSeriesPass(FunctionPass):
    """
    This pass converts DataFrame/Series operations to Array operations as much as
    possible to provide implementation and enable optimization. Creates specialized
    IR nodes for complex operations like Join.
    """

    _name = "bodo_series_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Convert Series after typing
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        series_pass = SeriesPass(
            state.func_ir,
            state.typingctx,
            state.targetctx,
            state.typemap,
            state.calltypes,
            state.locals,
            avoid_copy_propagation=state.pipeline.avoid_copy_propagation,
            parfor_metadata=state.metadata["parfors"],
        )
        # run multiple times to make sure transformations are fully applied
        # TODO(ehsan): run as long as IR changes
        orig_changed = series_pass.run()
        changed = orig_changed
        if changed:
            changed = series_pass.run()
        if changed:
            series_pass.run()
        return orig_changed


@register_pass(mutates_CFG=False, analysis_only=True)
class BodoDumpDistDiagnosticsPass(AnalysisPass):
    """Print Bodo's distributed diagnostics info if needed"""

    _name = "bodo_dump_diagnostics_pass"

    def __init__(self):
        AnalysisPass.__init__(self)

    def run_pass(self, state):
        """
        Print distributed diagnostics information if environment variable is
        set.
        """
        diag_level = 0
        env_name = "BODO_DISTRIBUTED_DIAGNOSTICS"
        try:
            diag_level = int(os.environ[env_name])
        except Exception:
            pass

        if (
            diag_level > 0 or state.flags.distributed_diagnostics
        ) and "distributed_diagnostics" in state.metadata:
            state.metadata["distributed_diagnostics"].dump(diag_level, state.metadata)
        return True


class BodoCompilerSeq(BodoCompiler):
    """Bodo pipeline without the distributed pass (used in rolling kernels)"""

    def define_pipelines(self):
        return self._create_bodo_pipeline(
            distributed=False, inline_calls_pass=inline_all_calls
        )


class BodoCompilerUDF(BodoCompiler):
    """Bodo pipeline with inlining and without the distributed pass (used in df.apply)"""

    def define_pipelines(self):
        return self._create_bodo_pipeline(distributed=False, udf_pipeline=True)


@register_pass(mutates_CFG=False, analysis_only=True)
class LowerParforSeq(FunctionPass):
    """Lower parfors to regular loops to avoid threading of Numba"""

    _name = "bodo_lower_parfor_seq_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        bodo.transforms.distributed_pass.lower_parfor_sequential(
            state.typingctx,
            state.func_ir,
            state.typemap,
            state.calltypes,
            state.metadata,
        )
        return True


@register_pass(mutates_CFG=False, analysis_only=True)
class LowerBodoIRExtSeq(FunctionPass):
    """Lower Bodo IR extensions nodes to regular Numba IR"""

    _name = "bodo_lower_ir_ext_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        from bodo.transforms.distributed_pass import distributed_run_extensions
        from bodo.transforms.table_column_del_pass import (
            remove_dead_table_columns,
        )
        from bodo.utils.transform import compile_func_single_block
        from bodo.utils.typing import (
            decode_if_dict_array,
            to_str_arr_if_dict_array,
        )

        state.func_ir._definitions = build_definitions(state.func_ir.blocks)

        # Run column pruning transformation from tables. This updates table source
        # nodes.
        typing_info = TypingInfo(
            state.typingctx,
            state.targetctx,
            state.typemap,
            state.calltypes,
            state.func_ir.loc,
        )
        remove_dead_table_columns(state.func_ir, state.typemap, typing_info)

        for block in state.func_ir.blocks.values():
            new_body = []
            for inst in block.body:
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
                    ):
                        out_nodes = f(
                            inst,
                            None,
                            state.typemap,
                            state.calltypes,
                            state.typingctx,
                            state.targetctx,
                            # is_independent -> Ranks will execute independently of
                            # other ranks due to bodo.jit(distributed=False),
                            # i.e., no collective operations should be used.
                            is_independent=True,
                            meta_head_only_info=None,
                        )
                    else:
                        out_nodes = f(
                            inst,
                            None,
                            state.typemap,
                            state.calltypes,
                            state.typingctx,
                            state.targetctx,
                        )
                    new_body += out_nodes
                elif is_call_assign(inst):
                    rhs = inst.value
                    fdef = guard(find_callname, state.func_ir, rhs)
                    # remove gatherv() in sequential mode to avoid hang
                    if fdef == ("gatherv", "bodo") or fdef == ("allgatherv", "bodo"):
                        lhs_typ = state.typemap[inst.target.name]
                        rhs_typ = state.typemap[rhs.args[0].name]
                        if (
                            # TODO: Can other types except arrays be read only.
                            isinstance(rhs_typ, types.Array)
                            and isinstance(lhs_typ, types.Array)
                        ):
                            modifiable_input = rhs_typ.copy(readonly=False)
                            modifiable_output = lhs_typ.copy(readonly=False)
                            if modifiable_input == modifiable_output:
                                # If data is equal make a copy.
                                new_body += compile_func_single_block(
                                    eval("lambda data: data.copy()"),
                                    (rhs.args[0],),
                                    inst.target,
                                    typing_info,
                                )
                                continue
                        if lhs_typ != rhs_typ and to_str_arr_if_dict_array(
                            lhs_typ
                        ) == to_str_arr_if_dict_array(rhs_typ):
                            new_body += compile_func_single_block(
                                eval("lambda data: decode_if_dict_array(data)"),
                                (rhs.args[0],),
                                inst.target,
                                typing_info,
                                extra_globals={
                                    "decode_if_dict_array": decode_if_dict_array
                                },
                            )
                            continue
                        else:
                            inst.value = rhs.args[0]

                    new_body.append(inst)
                else:
                    new_body.append(inst)

            block.body = new_body

        return True


@register_pass(mutates_CFG=False, analysis_only=True)
class BodoTableColumnDelPass(AnalysisPass):
    """Insert table column decref statements before distributed pass"""

    _name = "bodo_table_column_del_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        table_decref_pass = TableColumnDelPass(
            state.func_ir,
            state.typingctx,
            state.targetctx,
            state.typemap,
            state.calltypes,
        )
        return table_decref_pass.run()


class InlineValidationError(Exception):
    pass


def callee_ir_validator(func_ir):
    """Validates callee IR for unsupported features in inlining (currently only avoids
    functions with "with objmode" blocks)
    """
    for blk in func_ir.blocks.values():
        for stmt in blk.body:
            if isinstance(stmt, ir.EnterWith):
                raise InlineValidationError(
                    "callee_ir_validator: EnterWith not supported"
                )


def inline_calls(
    func_ir,
    _locals,
    work_list=None,
    typingctx=None,
    targetctx=None,
    typemap=None,
    calltypes=None,
):
    """Inlines all bodo.jit decorated functions in worklist.
    Returns the set of block labels that were processed.
    """
    if work_list is None:
        work_list = list(func_ir.blocks.items())
    new_labels = set()
    while work_list:
        label, block = work_list.pop()
        new_labels.add(label)
        for i, instr in enumerate(block.body):
            if isinstance(instr, ir.Assign):
                expr = instr.value
                if isinstance(expr, ir.Expr) and expr.op == "call":
                    func_def = guard(get_definition, func_ir, expr.func)

                    if (
                        isinstance(func_def, (ir.Global, ir.FreeVar))
                        and isinstance(func_def.value, CPUDispatcher)
                        and issubclass(
                            func_def.value._compiler.pipeline_class, BodoCompiler
                        )
                    ):
                        py_func = func_def.value.py_func
                        arg_types = None
                        # pass argument types if in a typed pass
                        if typingctx:
                            kws = dict(expr.kws)
                            a_types = tuple(typemap[v.name] for v in expr.args)
                            k_types = {k: typemap[v.name] for k, v in kws.items()}
                            _, arg_types = func_def.value.fold_argument_types(
                                a_types, k_types
                            )
                        try:
                            _, var_dict = inline_closure_call(
                                func_ir,
                                py_func.__globals__,
                                block,
                                i,
                                py_func,
                                typingctx=typingctx,
                                targetctx=targetctx,
                                arg_typs=arg_types,
                                typemap=typemap,
                                calltypes=calltypes,
                                work_list=work_list,
                                callee_validator=callee_ir_validator,
                            )
                        except InlineValidationError:
                            # Avoid inlining unsupported functions (e.g. has objmode)
                            continue

                        _locals.update(
                            (var_dict[k].name, v)
                            for k, v in func_def.value.locals.items()
                            if k in var_dict
                        )
                        # TODO: support options like "distributed" if applied to the
                        # inlined function

                        # current block is modified, skip the rest
                        # (included in new blocks)
                        break
    return new_labels


def udf_jit(signature_or_function=None, **options):
    """decorator for UDF implementation. Using Bodo's sequential/inline pipeline for
    the UDF to make sure nested calls are inlined and not distributed. Otherwise,
    generated barriers cause hangs. see: test_df_apply_func_case2
    """

    # wrap_python functions don't need to be compiled for UDFs
    if isinstance(signature_or_function, bodo.decorators.WrapPythonDispatcher):
        return signature_or_function

    parallel = {
        "comprehension": True,
        "setitem": False,
        # setting the new inplace_binop option to False until it is tested and handled
        # TODO: evaluate and enable
        "inplace_binop": False,
        "reduction": True,
        "numpy": True,
        # parallelizing stencils is not supported yet
        "stencil": False,
        "fusion": True,
    }
    return numba.njit(
        signature_or_function,
        parallel=parallel,
        pipeline_class=bodo.compiler.BodoCompilerUDF,
        **options,
    )


def is_udf_call(func_type):
    """deterimines if function type is a Bodo UDF call"""
    return (
        isinstance(func_type, numba.core.types.Dispatcher)
        and func_type.dispatcher._compiler.pipeline_class == BodoCompilerUDF
    )


def is_user_dispatcher(func_type):
    """determines if function type is a Bodo function written by
    a user rather than an internally written function."""
    # Func_type is a user function component if it is either from objmode or a
    # dispatcher with the BodoCompiler
    return (
        isinstance(func_type, numba.core.types.functions.ObjModeDispatcher)
        or isinstance(func_type, bodo.decorators.WrapPythonDispatcherType)
        or (
            isinstance(func_type, numba.core.types.Dispatcher)
            and issubclass(func_type.dispatcher._compiler.pipeline_class, BodoCompiler)
        )
    )


@register_pass(mutates_CFG=False, analysis_only=True)
class DummyCR(FunctionPass):
    """Dummy pass to add "cr" to compiler state to avoid errors in TyperCompiler since
    it doesn't have lowering.
    """

    _name = "bodo_dummy_cr"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        # save the compilation results to be used by get_func_type_info()
        # NOTE: Numba's "with lifting" (objmode handling) has a special structure where
        # it creates a new main function, compiles it, and returns its compilation
        # result instead of the original function. This results in the original
        # pipeline's data structures to be empty. Therefore we save compilation data
        # structures in our dummy pass which is the last compiler stage and called for
        # the new function, so the actual compiler data structures are available.
        state.cr = (
            state.func_ir,
            state.typemap,
            state.calltypes,
            state.return_type,
        )
        return True


def remove_passes_after(pm, location):
    """
    Remove all passes after `location` in PassManager's compilation pipeline
    """
    assert pm.passes
    pm._validate_pass(location)
    for idx, (x, _) in enumerate(pm.passes):
        if x == location:
            break
    else:  # pragma: no cover
        raise bodo.utils.typing.BodoError(f"Could not find pass {location}")
    pm.passes = pm.passes[: idx + 1]
    # if a pass has been added, it's not finalized
    pm._finalized = False


class TyperCompiler(BodoCompiler):
    """A compiler pipeline that skips passes after typing (provides typing info but not
    lowering).
    """

    def define_pipelines(self):
        [pm] = self._create_bodo_pipeline()
        # overload inlining is necessary since UDFs are inlined in series pass using
        # this pipeline
        # see test_spark_sql_array.py::test_array_repeat"[dataframe_val3]"
        remove_passes_after(pm, InlineOverloads)
        pm.add_pass_after(DummyCR, InlineOverloads)
        pm.finalize()
        return [pm]


def get_func_type_info(func, arg_types, kw_types):
    """
    Get IR and typing info for function 'func'. It creates a pipeline that runs all
    untyped passes as well as type inference.
    """
    # replicates Numba's pipeline initialization in these functions:
    # numba.core.dispatcher.Dispatcher.__init__ -> _compile_for_args -> compile
    # numba.core.dispatcher._FunctionCompiler.compile -> _compile_cached -> _compile_core
    # numba.core.compiler.compile_extra
    typingctx = numba.core.registry.cpu_target.typing_context
    targetctx = numba.core.registry.cpu_target.target_context
    library = None
    return_type = None
    _locals = {}
    pysig = numba.core.utils.pysignature(func)
    args = bodo.utils.transform.fold_argument_types(pysig, arg_types, kw_types)
    flags = numba.core.compiler.Flags()
    parallel_options = {
        "comprehension": True,
        "setitem": False,
        "inplace_binop": False,
        "reduction": True,
        "numpy": True,
        # parallelizing stencils is not supported yet
        "stencil": False,
        "fusion": True,
    }
    targetoptions = {
        "nopython": True,
        "boundscheck": False,
        "parallel": parallel_options,
    }
    numba.core.registry.cpu_target.options.parse_as_flags(flags, targetoptions)

    pipeline = TyperCompiler(
        typingctx, targetctx, library, args, return_type, flags, _locals
    )

    # DummyCR pass sets the compilation results as (func_ir, typemap, calltypes,
    # return_type)
    # NOTE: cannot use pipline.state.typemap and others since "with lifting" creates
    # a new main function and calls the compiler recursively, so pipeline doesn't have
    # the latest results. See test_groupby.py::test_groupby_apply_objmode
    return pipeline.compile_extra(func)
