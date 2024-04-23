# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Bodo's extension to object mode functionality.
"""

import warnings
from uuid import uuid4

import numba
from numba.core import errors, ir, ir_utils, types
from numba.core.transforms import find_region_inout_vars
from numba.core.withcontexts import (
    _clear_blocks,
    _mutate_with_block_caller,
)

import bodo
from bodo.utils.typing import BodoWarning

seen_functions = set()


def generate_objmode_warning(function_key: str):
    """Generates a Bodo warning if we have entered objmode and
    numba.core.config.DEVELOPER_MODE is enabled. We generate a unique key
    per objmode block to avoid duplicate warnings for streaming code.

    Args:
        function_key (str): A uuid4 unique key for the function.
    """
    global seen_functions
    if function_key not in seen_functions:
        if numba.core.config.DEVELOPER_MODE:
            warning_msg = (
                "Entered bodo.objmode. This will likely negatively impact performance."
            )
            warnings.warn(BodoWarning(warning_msg))
        seen_functions.add(function_key)


class _BodoObjModeContextType(numba.core.withcontexts._ObjModeContextType):
    def __init__(self, emit_warnings):
        super().__init__()
        self.emit_warnings = emit_warnings

    def mutate_with_body(
        self,
        func_ir,
        blocks,
        blk_start,
        blk_end,
        body_blocks,
        dispatcher_factory,
        extra,
    ):
        # Note: This is copied from numba except where Bodo Change is specified.
        cellnames = func_ir.func_id.func.__code__.co_freevars
        closures = func_ir.func_id.func.__closure__
        func_globals = func_ir.func_id.func.__globals__
        if closures is not None:
            # Resolve free variables
            func_closures = {}
            for cellname, closure in zip(cellnames, closures):
                try:
                    cellval = closure.cell_contents
                except ValueError as e:
                    # empty cell will raise
                    if str(e) != "Cell is empty":
                        raise
                else:
                    func_closures[cellname] = cellval
        else:
            # Missing closure object
            func_closures = {}
        args = extra["args"] if extra else ()
        kwargs = extra["kwargs"] if extra else {}

        typeanns = self._legalize_args(
            func_ir=func_ir,
            args=args,
            kwargs=kwargs,
            loc=blocks[blk_start].loc,
            func_globals=func_globals,
            func_closures=func_closures,
        )
        vlt = func_ir.variable_lifetime

        inputs, outputs = find_region_inout_vars(
            blocks=blocks,
            livemap=vlt.livemap,
            callfrom=blk_start,
            returnto=blk_end,
            body_block_ids=set(body_blocks),
        )

        # Determine types in the output tuple
        def strip_var_ver(x):
            return x.split(".", 1)[0]

        stripped_outs = list(map(strip_var_ver, outputs))

        # Verify that only outputs are annotated
        extra_annotated = set(typeanns) - set(stripped_outs)
        if extra_annotated:
            msg = (
                "Invalid type annotation on non-outgoing variables: {}."
                "Suggestion: remove annotation of the listed variables"
            )
            raise errors.TypingError(msg.format(extra_annotated))

        # Verify that all outputs are annotated

        # Note on "$cp" variable:
        # ``transforms.consolidate_multi_exit_withs()`` introduces the variable
        # for the control-point to determine the correct exit block. This
        # variable crosses the with-region boundary. Thus, it will be consider
        # an output variable leaving the lifted with-region.
        typeanns["$cp"] = types.int32
        not_annotated = set(stripped_outs) - set(typeanns)
        if not_annotated:
            msg = (
                "Missing type annotation on outgoing variable(s): {0}\n\n"
                "Example code: with objmode({1}='<"
                "add_type_as_string_here>')\n"
            )
            stable_ann = sorted(not_annotated)
            raise errors.TypingError(msg.format(stable_ann, stable_ann[0]))

        # Get output types
        outtup = types.Tuple([typeanns[v] for v in stripped_outs])

        lifted_blks = {k: blocks[k] for k in body_blocks}
        # Bodo Change: Indicate if we should emit warnings.
        _mutate_with_block_callee(
            lifted_blks,
            blk_start,
            blk_end,
            inputs,
            outputs,
            emit_warnings=self.emit_warnings,
        )

        lifted_ir = func_ir.derive(
            blocks=lifted_blks,
            arg_names=tuple(inputs),
            arg_count=len(inputs),
            force_non_generator=True,
        )
        dispatcher = dispatcher_factory(lifted_ir, objectmode=True, output_types=outtup)

        newblk = _mutate_with_block_caller(
            dispatcher,
            blocks,
            blk_start,
            blk_end,
            inputs,
            outputs,
        )

        blocks[blk_start] = newblk
        _clear_blocks(blocks, body_blocks)
        return dispatcher


def _mutate_with_block_callee(
    blocks, blk_start, blk_end, inputs, outputs, emit_warnings: bool
):
    """Mutate *blocks* for the callee of a with-context.

    Parameters
    ----------
    blocks : dict[ir.Block]
    blk_start, blk_end : int
        labels of the starting and ending block of the context-manager.
    inputs: sequence[str]
        Input variable names
    outputs: sequence[str]
        Output variable names
    """
    if not blocks:
        raise errors.NumbaValueError("No blocks in with-context block")
    head_blk = min(blocks)
    temp_blk = blocks[head_blk]
    scope = temp_blk.scope
    loc = temp_blk.loc

    # Bodo Change: Emit warnings in the prologue
    blocks[blk_start] = fill_callee_prologue(
        block=ir.Block(scope=scope, loc=loc),
        inputs=inputs,
        label_next=head_blk,
        emit_warnings=emit_warnings,
    )
    blocks[blk_end] = ir_utils.fill_callee_epilogue(
        block=ir.Block(scope=scope, loc=loc),
        outputs=outputs,
    )


def fill_callee_prologue(block, inputs, label_next, emit_warnings: bool):
    """
    Fill a new block *block* that unwraps arguments using names in *inputs* and
    then jumps to *label_next*.

    Expected to use with *fill_block_with_call()*
    """
    scope = block.scope
    loc = block.loc
    # load args
    args = [ir.Arg(name=k, index=i, loc=loc) for i, k in enumerate(inputs)]
    for aname, aval in zip(inputs, args):
        tmp = ir.Var(scope=scope, name=aname, loc=loc)
        block.append(ir.Assign(target=tmp, value=aval, loc=loc))
    # Bodo Change: Emit warnings in the prologue
    if emit_warnings:
        # Generate a unique function key so we only emit the warning once
        # for streaming.
        function_key = str(uuid4())
        function_key_var = ir.Var(
            scope,
            ir_utils.mk_unique_var("$function_key"),
            loc,
        )
        block.append(
            ir.Assign(
                target=function_key_var,
                value=ir.Const(value=function_key, loc=loc),
                loc=loc,
            )
        )
        # Create bodo.ir.object_mode.generate_objmode_warning
        bodo_var = ir.Var(
            scope,
            ir_utils.mk_unique_var("$bodo"),
            loc,
        )
        block.append(
            ir.Assign(
                target=bodo_var,
                value=ir.Global(name="bodo", value=bodo, loc=loc),
                loc=loc,
            )
        )
        ir_var = ir.Var(
            scope,
            ir_utils.mk_unique_var("$ir"),
            loc,
        )
        block.append(
            ir.Assign(
                target=ir_var,
                value=ir.Expr.getattr(value=bodo_var, attr="ir", loc=loc),
                loc=loc,
            )
        )
        object_mode_var = ir.Var(
            scope,
            ir_utils.mk_unique_var("$object_mode"),
            loc,
        )
        block.append(
            ir.Assign(
                target=object_mode_var,
                value=ir.Expr.getattr(value=ir_var, attr="object_mode", loc=loc),
                loc=loc,
            )
        )
        generate_objmode_warning_var = ir.Var(
            scope,
            ir_utils.mk_unique_var("$generate_objmode_warning"),
            loc,
        )
        block.append(
            ir.Assign(
                target=generate_objmode_warning_var,
                value=ir.Expr.getattr(
                    value=object_mode_var, attr="generate_objmode_warning", loc=loc
                ),
                loc=loc,
            )
        )
        # Generate the call
        warning_var = ir.Var(
            scope,
            ir_utils.mk_unique_var("$warning"),
            loc,
        )
        block.append(
            ir.Assign(
                target=warning_var,
                value=ir.Expr.call(
                    func=generate_objmode_warning_var,
                    args=[function_key_var],
                    kws=[],
                    loc=loc,
                ),
                loc=loc,
            )
        )
    # jump to loop entry
    block.append(ir.Jump(target=label_next, loc=loc))
    return block


warning_objmode = _BodoObjModeContextType(emit_warnings=True)
no_warning_objmode = _BodoObjModeContextType(emit_warnings=False)
