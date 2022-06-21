"""
Numba monkey patches to fix issues related to Bodo. Should be imported before any
other module in bodo package.
"""
import copy
import functools
import hashlib
import inspect
import itertools
import operator
import os
import re
import sys
import textwrap
import traceback
import types as pytypes
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import ExitStack

import numba
import numba.core.boxing
import numba.core.inline_closurecall
import numba.core.typing.listdecl
import numba.np.linalg
from numba.core import analysis, cgutils, errors, ir, ir_utils, types
from numba.core.compiler import Compiler
from numba.core.errors import ForceLiteralArg, LiteralTypingError, TypingError
from numba.core.ir_utils import (
    GuardException,
    _create_function_from_code_obj,
    analysis,
    build_definitions,
    find_callname,
    get_definition,
    guard,
    has_no_side_effect,
    mk_unique_var,
    remove_dead_extensions,
    replace_vars_inner,
    require,
    visit_vars_extensions,
    visit_vars_inner,
)
from numba.core.types import literal
from numba.core.types.functions import (
    _bt_as_lines,
    _ResolutionFailures,
    _termcolor,
    _unlit_non_poison,
)
from numba.core.typing.templates import (
    AbstractTemplate,
    Signature,
    _EmptyImplementationEntry,
    _inline_info,
    _OverloadAttributeTemplate,
    infer_global,
    signature,
)
from numba.core.typing.typeof import Purpose, typeof
from numba.experimental.jitclass import base as jitclass_base
from numba.experimental.jitclass import decorators as jitclass_decorators
from numba.extending import NativeValue, lower_builtin, typeof_impl
from numba.parfors.parfor import get_expr_args

from bodo.utils.python_310_bytecode_pass import (
    Bodo310ByteCodePass,
    peep_hole_call_function_ex_to_call_function_kw,
    peep_hole_fuse_dict_add_updates,
)
from bodo.utils.typing import (
    BodoError,
    get_overload_const_str,
    is_overload_constant_str,
    raise_bodo_error,
)

# flag for checking whether the functions we are replacing have changed in a later Numba
# release. Needs to be checked for every new Numba release so we update our changes.
_check_numba_change = False


# Make sure literals are tried first for typing Bodo's intrinsics, since output type
# may depend on literals.
# see test_join.py::test_merge_index_column_second"[df21-df10]"
numba.core.typing.templates._IntrinsicTemplate.prefer_literal = True


# `run_frontend` function of Numba is used in inline_closure_call to get the IR of the
# function to be inlined.
# The code below is copied from Numba and modified to handle 'raise' nodes by running
# rewrite passes before inlining (feature copied from numba.core.ir_utils.get_ir_of_code).
# usecase example: bodo/tests/test_series.py::test_series_combine"[S13-S23-None-True]"
# https://github.com/numba/numba/blob/cc7e7c7cfa6389b54d3b5c2c95751c97eb531a96/numba/compiler.py#L186
def run_frontend(func, inline_closures=False, emit_dels=False):
    """
    Run the compiler frontend over the given Python function, and return
    the function's canonical Numba IR.

    If inline_closures is Truthy then closure inlining will be run
    If emit_dels is Truthy the ir.Del nodes will be emitted appropriately
    """
    from numba.core.utils import PYVERSION

    # XXX make this a dedicated Pipeline?
    func_id = numba.core.bytecode.FunctionIdentity.from_function(func)
    interp = numba.core.interpreter.Interpreter(func_id)
    bc = numba.core.bytecode.ByteCode(func_id=func_id)
    func_ir = interp.interpret(bc)
    # BODO Change: add 3.10 byte code changes until Numba 0.56 if PYVERSION is 3.10
    if PYVERSION == (3, 10):
        func_ir = peep_hole_call_function_ex_to_call_function_kw(func_ir)
        func_ir = peep_hole_fuse_dict_add_updates(func_ir)

    if inline_closures:
        from numba.core.inline_closurecall import InlineClosureCallPass

        # code added to original 'run_frontend' to add rewrite passes
        # we need to run the before inference rewrite pass to normalize the IR
        # XXX: check rewrite pass flag?
        # for example, Raise nodes need to become StaticRaise before type inference
        class DummyPipeline:
            def __init__(self, f_ir):
                self.state = numba.core.compiler.StateDict()
                self.state.typingctx = None
                self.state.targetctx = None
                self.state.args = None
                self.state.func_ir = f_ir
                self.state.typemap = None
                self.state.return_type = None
                self.state.calltypes = None

        numba.core.rewrites.rewrite_registry.apply(
            "before-inference", DummyPipeline(func_ir).state
        )
        inline_pass = InlineClosureCallPass(
            func_ir, numba.core.cpu.ParallelOptions(False), {}, False
        )
        inline_pass.run()
    post_proc = numba.core.postproc.PostProcessor(func_ir)
    post_proc.run(emit_dels)
    return func_ir


if _check_numba_change:  # pragma: no cover
    # make sure run_frontend hasn't changed before replacing it
    lines = inspect.getsource(numba.core.compiler.run_frontend)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "8c2477a793b2c08d56430997880974ac12c5570e69c9e54d37d694b322ea18b6"
    ):  # pragma: no cover
        warnings.warn("numba.core.compiler.run_frontend has changed")


numba.core.compiler.run_frontend = run_frontend


# replace visit_vars_stmt of Numba to handle vararg attribute of Print nodes
def visit_vars_stmt(stmt, callback, cbdata):
    # let external calls handle stmt if type matches
    for t, f in visit_vars_extensions.items():
        if isinstance(stmt, t):
            f(stmt, callback, cbdata)
            return
    if isinstance(stmt, ir.Assign):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.value = visit_vars_inner(stmt.value, callback, cbdata)
    elif isinstance(stmt, ir.Arg):
        stmt.name = visit_vars_inner(stmt.name, callback, cbdata)
    elif isinstance(stmt, ir.Return):
        stmt.value = visit_vars_inner(stmt.value, callback, cbdata)
    elif isinstance(stmt, ir.Raise):
        stmt.exception = visit_vars_inner(stmt.exception, callback, cbdata)
    elif isinstance(stmt, ir.Branch):
        stmt.cond = visit_vars_inner(stmt.cond, callback, cbdata)
    elif isinstance(stmt, ir.Jump):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
    elif isinstance(stmt, ir.Del):
        # Because Del takes only a var name, we make up by
        # constructing a temporary variable.
        var = ir.Var(None, stmt.value, stmt.loc)
        var = visit_vars_inner(var, callback, cbdata)
        stmt.value = var.name
    elif isinstance(stmt, ir.DelAttr):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.attr = visit_vars_inner(stmt.attr, callback, cbdata)
    elif isinstance(stmt, ir.SetAttr):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.attr = visit_vars_inner(stmt.attr, callback, cbdata)
        stmt.value = visit_vars_inner(stmt.value, callback, cbdata)
    elif isinstance(stmt, ir.DelItem):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.index = visit_vars_inner(stmt.index, callback, cbdata)
    elif isinstance(stmt, ir.StaticSetItem):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.index_var = visit_vars_inner(stmt.index_var, callback, cbdata)
        stmt.value = visit_vars_inner(stmt.value, callback, cbdata)
    elif isinstance(stmt, ir.SetItem):
        stmt.target = visit_vars_inner(stmt.target, callback, cbdata)
        stmt.index = visit_vars_inner(stmt.index, callback, cbdata)
        stmt.value = visit_vars_inner(stmt.value, callback, cbdata)
    elif isinstance(stmt, ir.Print):
        stmt.args = [visit_vars_inner(x, callback, cbdata) for x in stmt.args]
        # Bodo change: support vararg for Print nodes
        stmt.vararg = visit_vars_inner(stmt.vararg, callback, cbdata)
    else:
        # TODO: raise NotImplementedError("no replacement for IR node: ", stmt)
        pass
    return


if _check_numba_change:  # pragma: no cover
    # make sure visit_vars_stmt hasn't changed before replacing it
    lines = inspect.getsource(numba.core.ir_utils.visit_vars_stmt)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "52b7b645ba65c35f3cf564f936e113261db16a2dff1e80fbee2459af58844117"
    ):  # pragma: no cover
        warnings.warn("numba.core.ir_utils.visit_vars_stmt has changed")


numba.core.ir_utils.visit_vars_stmt = visit_vars_stmt


old_run_pass = numba.core.typed_passes.InlineOverloads.run_pass


def InlineOverloads_run_pass(self, state):
    """plug in Bodo's overload inliner in Numba overload inliner to accelerate
    compilation time (e.g. single block functions are faster to inline).

    Plugging in existing inliner instead of a new pass since Numba overload
    implementations are compiled recursively, so our inliner should be part of regular
    Numba pipeline.
    """
    import bodo

    bodo.compiler.bodo_overload_inline_pass(
        state.func_ir, state.typingctx, state.targetctx, state.typemap, state.calltypes
    )
    return old_run_pass(self, state)


numba.core.typed_passes.InlineOverloads.run_pass = InlineOverloads_run_pass


# The code below is copied from Numba and modified to handle aliases with tuple values.
# https://github.com/numba/numba/blob/cc7e7c7cfa6389b54d3b5c2c95751c97eb531a96/numba/ir_utils.py#L725
# This case happens for Bodo dataframes since init_dataframe takes a tuple of arrays as
# input, and output dataframe is aliased with all of these arrays. see test_df_alias.
from numba.core.ir_utils import (
    _add_alias,
    alias_analysis_extensions,
    alias_func_extensions,
)

# immutable scalar types, no aliasing possible
_immutable_type_class = (
    types.Number,
    types.scalars._NPDatetimeBase,
    types.iterators.RangeType,
    types.UnicodeType,
)


def is_immutable_type(var, typemap):
    # Conservatively, assume mutable if type not available
    if typemap is None or var not in typemap:
        return False
    typ = typemap[var]

    # TODO: add more immutable types
    if isinstance(typ, _immutable_type_class):
        return True

    if isinstance(typ, types.BaseTuple) and all(
        isinstance(t, _immutable_type_class) for t in typ.types
    ):
        return True
    # consevatively, assume mutable
    return False


def find_potential_aliases(
    blocks, args, typemap, func_ir, alias_map=None, arg_aliases=None
):
    "find all array aliases and argument aliases to avoid remove as dead"
    if alias_map is None:
        alias_map = {}
    if arg_aliases is None:
        arg_aliases = set(a for a in args if not is_immutable_type(a, typemap))

    # update definitions since they are not guaranteed to be up-to-date
    # FIXME keep definitions up-to-date to avoid the need for rebuilding
    func_ir._definitions = build_definitions(func_ir.blocks)
    np_alias_funcs = ["ravel", "transpose", "reshape"]

    for bl in blocks.values():
        for instr in bl.body:
            if type(instr) in alias_analysis_extensions:
                f = alias_analysis_extensions[type(instr)]
                f(instr, args, typemap, func_ir, alias_map, arg_aliases)
            if isinstance(instr, ir.Assign):
                expr = instr.value
                lhs = instr.target.name
                # only mutable types can alias
                if is_immutable_type(lhs, typemap):
                    continue
                if isinstance(expr, ir.Var) and lhs != expr.name:
                    _add_alias(lhs, expr.name, alias_map, arg_aliases)
                # subarrays like A = B[0] for 2D B
                if isinstance(expr, ir.Expr) and (
                    expr.op == "cast" or expr.op in ["getitem", "static_getitem"]
                ):
                    _add_alias(lhs, expr.value.name, alias_map, arg_aliases)
                if isinstance(expr, ir.Expr) and expr.op == "inplace_binop":
                    _add_alias(lhs, expr.lhs.name, alias_map, arg_aliases)
                # array attributes like A.T
                if (
                    isinstance(expr, ir.Expr)
                    and expr.op == "getattr"
                    and expr.attr in ["T", "ctypes", "flat"]
                ):
                    _add_alias(lhs, expr.value.name, alias_map, arg_aliases)
                # a = b.c.  a should alias b
                if (
                    isinstance(expr, ir.Expr)
                    and expr.op == "getattr"
                    and expr.attr not in ["shape"]
                    and expr.value.name in arg_aliases
                ):
                    _add_alias(lhs, expr.value.name, alias_map, arg_aliases)
                # Bodo change: handle potential Series, DataFrame, ... aliases.
                # Types may not be available yet but type check is not necessary since
                # extra aliases are ok.
                if (
                    isinstance(expr, ir.Expr)
                    and expr.op == "getattr"
                    and expr.attr
                    in ("loc", "iloc", "iat", "_obj", "obj", "codes", "_df")
                ):
                    _add_alias(lhs, expr.value.name, alias_map, arg_aliases)
                # new code added to handle tuple/list/set of mutable data
                if (
                    isinstance(expr, ir.Expr)
                    and expr.op in ("build_tuple", "build_list", "build_set")
                    and not is_immutable_type(lhs, typemap)
                ):
                    for v in expr.items:
                        _add_alias(lhs, v.name, alias_map, arg_aliases)
                # calls that can create aliases such as B = A.ravel()
                if isinstance(expr, ir.Expr) and expr.op == "call":
                    fdef = guard(find_callname, func_ir, expr, typemap)
                    # TODO: sometimes gufunc backend creates duplicate code
                    # causing find_callname to fail. Example: test_argmax
                    # ignored here since those cases don't create aliases
                    # but should be fixed in general
                    if fdef is None:
                        continue
                    fname, fmod = fdef
                    if fdef in alias_func_extensions:
                        alias_func = alias_func_extensions[fdef]
                        alias_func(lhs, expr.args, alias_map, arg_aliases)
                    if fmod == "numpy" and fname in np_alias_funcs:
                        _add_alias(lhs, expr.args[0].name, alias_map, arg_aliases)
                    if isinstance(fmod, ir.Var) and fname in np_alias_funcs:
                        _add_alias(lhs, fmod.name, alias_map, arg_aliases)

    # copy to avoid changing size during iteration
    old_alias_map = copy.deepcopy(alias_map)
    # combine all aliases transitively
    for v in old_alias_map:
        for w in old_alias_map[v]:
            alias_map[v] |= alias_map[w]
        for w in old_alias_map[v]:
            alias_map[w] = alias_map[v]

    return alias_map, arg_aliases


if _check_numba_change:  # pragma: no cover
    # make sure find_potential_aliases hasn't changed before replacing it
    lines = inspect.getsource(ir_utils.find_potential_aliases)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "e6cf3e0f502f903453eb98346fc6854f87dc4ea1ac62f65c2d6aef3bf690b6c5"
    ):  # pragma: no cover
        warnings.warn("ir_utils.find_potential_aliases has changed")


ir_utils.find_potential_aliases = find_potential_aliases
# This is also imported in array analysis
numba.parfors.array_analysis.find_potential_aliases = find_potential_aliases


if _check_numba_change:  # pragma: no cover
    # make sure dead_code_elimination hasn't changed before replacing it
    lines = inspect.getsource(ir_utils.dead_code_elimination)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "40a8626300a1a17523944ec7842b093c91258bbc60844bbd72191a35a4c366bf"
    ):  # pragma: no cover
        warnings.warn("ir_utils.dead_code_elimination has changed")

# replace dead_code_elimination function with a mini version since it is not safe for
# Numba passes before our SeriesPass (currently InlineOverloads/InlineClosureCallPass)
# to run dead code elimination. Alias analysis does not know about DataFrame/Series
# aliases like Series.loc yet.
# TODO(ehsan): add DataFrame/Series aliases to alias analysis
def mini_dce(func_ir, typemap=None, alias_map=None, arg_aliases=None):
    """A mini dead code elimination function that is similar to ir_utils.remove_dead()
    but is much more conservative. Only removes nodes if node aliasing or side effect
    is not possible.
    Required for InlineClosureCallPass since some of the leftover make_function nodes
    cannot be handled later (see test_join_string for example).
    """
    from numba.core.analysis import (
        compute_cfg_from_blocks,
        compute_live_map,
        compute_use_defs,
    )

    cfg = compute_cfg_from_blocks(func_ir.blocks)
    usedefs = compute_use_defs(func_ir.blocks)
    live_map = compute_live_map(cfg, func_ir.blocks, usedefs.usemap, usedefs.defmap)

    # repeat until convergence
    changed = True

    while changed:
        changed = False
        for label, block in func_ir.blocks.items():
            # find live variables at each statement to delete dead assignment
            lives = {v.name for v in block.terminator.list_vars()}
            # find live variables at the end of block
            for out_blk, _data in cfg.successors(label):
                lives |= live_map[out_blk]

            new_body = [block.terminator]
            # for each statement in reverse order, excluding terminator
            for stmt in reversed(block.body[:-1]):

                # ignore assignments that their lhs is not live or lhs==rhs
                if isinstance(stmt, ir.Assign):
                    lhs = stmt.target
                    rhs = stmt.value
                    if lhs.name not in lives:
                        # make_function nodes are always safe to remove since they don't
                        # introduce any aliases and have no side effects
                        if isinstance(rhs, ir.Expr) and rhs.op == "make_function":
                            continue
                        # getattr doesn't have any side effects
                        if isinstance(rhs, ir.Expr) and rhs.op == "getattr":
                            continue
                        # Const values are safe to remove since alias is not possible
                        if isinstance(rhs, ir.Const):
                            continue
                        # Function values are safe to remove since aliasing not possible
                        if typemap and isinstance(
                            typemap.get(lhs, None), types.Function
                        ):
                            continue
                        # build_map doesn't have any side effects
                        if isinstance(rhs, ir.Expr) and rhs.op == "build_map":
                            continue
                        # build_tuple doesn't have any side effects
                        if isinstance(rhs, ir.Expr) and rhs.op == "build_tuple":
                            continue
                    if isinstance(rhs, ir.Var) and lhs.name == rhs.name:
                        continue

                # Del nodes are safe to remove since there is no side effect
                if isinstance(stmt, ir.Del):
                    if stmt.value not in lives:
                        continue

                if type(stmt) in analysis.ir_extension_usedefs:
                    def_func = analysis.ir_extension_usedefs[type(stmt)]
                    uses, defs = def_func(stmt)
                    lives -= defs
                    lives |= uses
                else:
                    lives |= {v.name for v in stmt.list_vars()}
                    if isinstance(stmt, ir.Assign):
                        lives.remove(lhs.name)

                new_body.append(stmt)
            new_body.reverse()
            if len(block.body) != len(new_body):
                changed = True
            block.body = new_body


ir_utils.dead_code_elimination = mini_dce
numba.core.typed_passes.dead_code_elimination = mini_dce
numba.core.inline_closurecall.dead_code_elimination = mini_dce


# replace Numba's make_overload_template to support a new option
# called 'no_unliteral', which avoids a second run of overload with literal types
# converted to non-literal versions. This solves hiding errors such as #889
# TODO: remove after Numba's #5411 is resolved

from numba.core.cpu_options import InlineOptions


# change: added no_unliteral argument
def make_overload_template(
    func, overload_func, jit_options, strict, inline, prefer_literal=False, **kwargs
):
    """
    Make a template class for function *func* overloaded by *overload_func*.
    Compiler options are passed as a dictionary to *jit_options*.
    """
    func_name = getattr(func, "__name__", str(func))
    name = "OverloadTemplate_%s" % (func_name,)
    # Bodo change: added no_unliteral argument
    no_unliteral = kwargs.pop("no_unliteral", False)
    base = numba.core.typing.templates._OverloadFunctionTemplate
    dct = dict(
        key=func,
        _overload_func=staticmethod(overload_func),
        _impl_cache={},
        _compiled_overloads={},
        _jit_options=jit_options,
        _strict=strict,
        _inline=staticmethod(InlineOptions(inline)),
        _inline_overloads={},
        prefer_literal=prefer_literal,
        # Bodo change: added no_unliteral argument
        _no_unliteral=no_unliteral,
        metadata=kwargs,
    )
    return type(base)(name, (base,), dct)


if _check_numba_change:  # pragma: no cover
    # make sure make_overload_template hasn't changed before replacing it
    lines = inspect.getsource(numba.core.typing.templates.make_overload_template)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "7f6974584cb10e49995b652827540cc6732e497c0b9f8231b44fd83fcc1c0a83"
    ):  # pragma: no cover
        warnings.warn("numba.core.typing.templates.make_overload_template has changed")


numba.core.typing.templates.make_overload_template = make_overload_template


def _resolve(self, typ, attr):
    if self._attr != attr:
        return None

    if isinstance(typ, types.TypeRef):
        assert typ == self.key
    else:
        assert isinstance(typ, self.key)

    class MethodTemplate(AbstractTemplate):
        key = (self.key, attr)
        _inline = self._inline
        # Bodo change: added _no_unliteral attribute
        _no_unliteral = getattr(self, "_no_unliteral", False)
        _overload_func = staticmethod(self._overload_func)
        _inline_overloads = self._inline_overloads
        prefer_literal = self.prefer_literal

        def generic(_, args, kws):
            args = (typ,) + tuple(args)
            fnty = self._get_function_type(self.context, typ)
            sig = self._get_signature(self.context, fnty, args, kws)
            sig = sig.replace(pysig=numba.core.utils.pysignature(self._overload_func))
            for template in fnty.templates:
                self._inline_overloads.update(template._inline_overloads)
            if sig is not None:
                return sig.as_method()

    return types.BoundFunction(MethodTemplate, typ)


if _check_numba_change:  # pragma: no cover
    # make sure _resolve hasn't changed before replacing it
    lines = inspect.getsource(
        numba.core.typing.templates._OverloadMethodTemplate._resolve
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "ce8e0935dc939d0867ef969e1ed2975adb3533a58a4133fcc90ae13c4418e4d6"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.typing.templates._OverloadMethodTemplate._resolve has changed"
        )


numba.core.typing.templates._OverloadMethodTemplate._resolve = _resolve


# change: added no_unliteral argument
def make_overload_attribute_template(
    typ,
    attr,
    overload_func,
    inline,
    prefer_literal=False,
    base=_OverloadAttributeTemplate,
    **kwargs,
):
    """
    Make a template class for attribute *attr* of *typ* overloaded by
    *overload_func*.
    """
    assert isinstance(typ, types.Type) or issubclass(typ, types.Type)
    name = "OverloadAttributeTemplate_%s_%s" % (typ, attr)
    # Bodo change: added _no_unliteral attribute
    no_unliteral = kwargs.pop("no_unliteral", False)
    # Note the implementation cache is subclass-specific
    dct = dict(
        key=typ,
        _attr=attr,
        _impl_cache={},
        _inline=staticmethod(InlineOptions(inline)),
        _inline_overloads={},
        # Bodo change: added _no_unliteral argument
        _no_unliteral=no_unliteral,
        _overload_func=staticmethod(overload_func),
        prefer_literal=prefer_literal,
        metadata=kwargs,
    )
    obj = type(base)(name, (base,), dct)
    return obj


if _check_numba_change:  # pragma: no cover
    # make sure make_overload_attribute_template hasn't changed before replacing it
    lines = inspect.getsource(
        numba.core.typing.templates.make_overload_attribute_template
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "f066c38c482d6cf8bf5735a529c3264118ba9b52264b24e58aad12a6b1960f5d"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.typing.templates.make_overload_attribute_template has changed"
        )


numba.core.typing.templates.make_overload_attribute_template = (
    make_overload_attribute_template
)


# replace overload inline handling to avoid recompilation
def generic(self, args, kws):
    """
    Type the overloaded function by compiling the appropriate
    implementation for the given args.
    """
    from numba.core.typed_passes import PreLowerStripPhis

    disp, new_args = self._get_impl(args, kws)
    if disp is None:
        return
    # Compile and type it for the given types
    disp_type = types.Dispatcher(disp)
    # Store the compiled overload for use in the lowering phase if there's
    # no inlining required (else functions are being compiled which will
    # never be used as they are inlined)
    if not self._inline.is_never_inline:
        # need to run the compiler front end up to type inference to compute
        # a signature
        from numba.core import compiler, typed_passes
        from numba.core.inline_closurecall import InlineWorker

        fcomp = disp._compiler
        flags = compiler.Flags()

        # Updating these causes problems?!
        # fcomp.targetdescr.options.parse_as_flags(flags,
        #                                         fcomp.targetoptions)
        # flags = fcomp._customize_flags(flags)

        # spoof a compiler pipline like the one that will be in use
        tyctx = fcomp.targetdescr.typing_context
        tgctx = fcomp.targetdescr.target_context
        compiler_inst = fcomp.pipeline_class(
            tyctx,
            tgctx,
            None,
            None,
            None,
            flags,
            None,
        )
        inline_worker = InlineWorker(
            tyctx,
            tgctx,
            fcomp.locals,
            compiler_inst,
            flags,
            None,
        )

        # If the inlinee contains something to trigger literal arg dispatch
        # then the pipeline call will unconditionally fail due to a raised
        # ForceLiteralArg exception. Therefore `resolve` is run first, as
        # type resolution must occur at some point, this will hit any
        # `literally` calls and because it's going via the dispatcher will
        # handle them correctly i.e. ForceLiteralArg propagates. This having
        # the desired effect of ensuring the pipeline call is only made in
        # situations that will succeed. For context see #5887.
        resolve = disp_type.dispatcher.get_call_template
        template, pysig, folded_args, kws = resolve(new_args, kws)

        # Bodo change:
        # avoid recompiling the implementation if info already available
        if folded_args in self._inline_overloads:
            return self._inline_overloads[folded_args]["iinfo"].signature

        ir = inline_worker.run_untyped_passes(
            disp_type.dispatcher.py_func, enable_ssa=True
        )

        (typemap, return_type, calltypes, _) = typed_passes.type_inference_stage(
            self.context, tgctx, ir, folded_args, None
        )
        ir = PreLowerStripPhis()._strip_phi_nodes(ir)
        ir._definitions = numba.core.ir_utils.build_definitions(ir.blocks)

        sig = Signature(return_type, folded_args, None)
        # this stores a load of info for the cost model function if supplied
        # it by default is None
        self._inline_overloads[sig.args] = {"folded_args": folded_args}
        # this stores the compiled overloads, if there's no compiled
        # overload available i.e. function is always inlined, the key still
        # needs to exist for type resolution

        # NOTE: If lowering is failing on a `_EmptyImplementationEntry`,
        #       the inliner has failed to inline this entry corretly.
        impl_init = _EmptyImplementationEntry("always inlined")
        self._compiled_overloads[sig.args] = impl_init
        if not self._inline.is_always_inline:
            # this branch is here because a user has supplied a function to
            # determine whether to inline or not. As a result both compiled
            # function and inliner info needed, delaying the computation of
            # this leads to an internal state mess at present. TODO: Fix!
            sig = disp_type.get_call_type(self.context, new_args, kws)
            self._compiled_overloads[sig.args] = disp_type.get_overload(sig)
            # store the inliner information, it's used later in the cost
            # model function call
        iinfo = _inline_info(ir, typemap, calltypes, sig)
        self._inline_overloads[sig.args] = {"folded_args": folded_args, "iinfo": iinfo}
    else:
        sig = disp_type.get_call_type(self.context, new_args, kws)
        if sig is None:  # can't resolve for this target
            return None
        self._compiled_overloads[sig.args] = disp_type.get_overload(sig)
    return sig


if _check_numba_change:  # pragma: no cover
    # make sure generic() hasn't changed before replacing it
    lines = inspect.getsource(
        numba.core.typing.templates._OverloadFunctionTemplate.generic
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "5d453a6d0215ebf0bab1279ff59eb0040b34938623be99142ce20acc09cdeb64"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.typing.templates._OverloadFunctionTemplate.generic has changed"
        )


numba.core.typing.templates._OverloadFunctionTemplate.generic = generic


def bound_function(template_key, no_unliteral=False):
    """
    Wrap an AttributeTemplate resolve_* method to allow it to
    resolve an instance method's signature rather than a instance attribute.
    The wrapped method must return the resolved method's signature
    according to the given self type, args, and keywords.

    It is used thusly:

        class ComplexAttributes(AttributeTemplate):
            @bound_function("complex.conjugate")
            def resolve_conjugate(self, ty, args, kwds):
                return ty

    *template_key* (e.g. "complex.conjugate" above) will be used by the
    target to look up the method's implementation, as a regular function.
    """

    def wrapper(method_resolver):
        @functools.wraps(method_resolver)
        def attribute_resolver(self, ty):
            class MethodTemplate(AbstractTemplate):
                key = template_key

                def generic(_, args, kws):
                    sig = method_resolver(self, ty, args, kws)
                    if sig is not None and sig.recvr is None:
                        sig = sig.replace(recvr=ty)
                    return sig

            # bodo change: adding no_unliteral flag
            MethodTemplate._no_unliteral = no_unliteral
            return types.BoundFunction(MethodTemplate, ty)

        return attribute_resolver

    return wrapper


if _check_numba_change:  # pragma: no cover
    # make sure bound_function hasn't changed before replacing it
    lines = inspect.getsource(numba.core.typing.templates.bound_function)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "a2feefe64eae6a15c56affc47bf0c1d04461f9566913442d539452b397103322"
    ):  # pragma: no cover
        warnings.warn("numba.core.typing.templates.bound_function has changed")


numba.core.typing.templates.bound_function = bound_function


def get_call_type(self, context, args, kws):
    from numba.core import utils

    prefer_lit = [True, False]  # old behavior preferring literal
    prefer_not = [False, True]  # new behavior preferring non-literal
    failures = _ResolutionFailures(context, self, args, kws, depth=self._depth)

    # BODO TODO: does target handling increase compilation time?
    # get the order in which to try templates
    from numba.core.target_extension import get_local_target  # circular

    target_hw = get_local_target(context)
    order = utils.order_by_target_specificity(
        target_hw, self.templates, fnkey=self.key[0]
    )

    self._depth += 1
    for temp_cls in order:
        temp = temp_cls(context)
        # The template can override the default and prefer literal args
        choice = prefer_lit if temp.prefer_literal else prefer_not
        # Bodo change: check _no_unliteral attribute if present
        choice = [True] if getattr(temp, "_no_unliteral", False) else choice
        for uselit in choice:
            try:
                if uselit:
                    sig = temp.apply(args, kws)
                else:
                    nolitargs = tuple([_unlit_non_poison(a) for a in args])
                    nolitkws = {k: _unlit_non_poison(v) for k, v in kws.items()}
                    sig = temp.apply(nolitargs, nolitkws)
            except Exception as e:
                from numba.core import utils

                if utils.use_new_style_errors() and not isinstance(
                    e, errors.NumbaError
                ):
                    raise e
                else:
                    sig = None
                    failures.add_error(temp, False, e, uselit)
            else:
                if sig is not None:
                    self._impl_keys[sig.args] = temp.get_impl_key(sig)
                    self._depth -= 1
                    return sig
                else:
                    registered_sigs = getattr(temp, "cases", None)
                    if registered_sigs is not None:
                        msg = "No match for registered cases:\n%s"
                        msg = msg % "\n".join(
                            " * {}".format(x) for x in registered_sigs
                        )
                    else:
                        msg = "No match."
                    failures.add_error(temp, True, msg, uselit)

    failures.raise_error()


if _check_numba_change:  # pragma: no cover
    # make sure get_call_type hasn't changed before replacing it
    lines = inspect.getsource(numba.core.types.functions.BaseFunction.get_call_type)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "25f038a7216f8e6f40068ea81e11fd9af8ad25d19888f7304a549941b01b7015"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.types.functions.BaseFunction.get_call_type has changed"
        )


numba.core.types.functions.BaseFunction.get_call_type = get_call_type

bodo_typing_error_info = """
This is often caused by the use of unsupported features or typing issues.
See https://docs.bodo.ai/
"""


def get_call_type2(self, context, args, kws):
    template = self.template(context)
    literal_e = None
    nonliteral_e = None
    out = None

    choice = [True, False] if template.prefer_literal else [False, True]
    # Bodo change: check _no_unliteral attribute if present
    choice = [True] if getattr(template, "_no_unliteral", False) else choice
    for uselit in choice:
        if uselit:
            # Try with Literal
            try:
                out = template.apply(args, kws)
            except Exception as exc:
                if isinstance(exc, errors.ForceLiteralArg):
                    raise exc
                literal_e = exc
                out = None
            else:
                break
        else:
            # if the unliteral_args and unliteral_kws are the same as the literal
            # ones, set up to not bother retrying
            unliteral_args = tuple([_unlit_non_poison(a) for a in args])
            unliteral_kws = {k: _unlit_non_poison(v) for k, v in kws.items()}
            skip = unliteral_args == args and kws == unliteral_kws

            # If the above template application failed and the non-literal args are
            # different to the literal ones, try again with literals rewritten as
            # non-literals
            if not skip and out is None:
                try:
                    out = template.apply(unliteral_args, unliteral_kws)
                except Exception as exc:
                    from numba.core import utils

                    if utils.use_new_style_errors() and not isinstance(
                        exc, errors.NumbaError
                    ):
                        raise exc
                    if isinstance(exc, errors.ForceLiteralArg):
                        if template.prefer_literal:
                            # For template that prefers literal types,
                            # reaching here means that the literal types
                            # have failed typing as well.
                            raise exc
                    nonliteral_e = exc
                else:
                    break

    if out is None and (nonliteral_e is not None or literal_e is not None):
        header = "- Resolution failure for {} arguments:\n{}\n"
        tmplt = _termcolor.highlight(header)
        if numba.core.config.DEVELOPER_MODE:
            indent = " " * 4

            def add_bt(error):
                if isinstance(error, BaseException):
                    # if the error is an actual exception instance, trace it
                    bt = traceback.format_exception(
                        type(error), error, error.__traceback__
                    )
                else:
                    bt = [""]
                nd2indent = "\n{}".format(2 * indent)
                errstr = _termcolor.reset(nd2indent + nd2indent.join(_bt_as_lines(bt)))
                return _termcolor.reset(errstr)

        else:
            add_bt = lambda X: ""

        def nested_msg(literalness, e):
            estr = str(e)
            estr = estr if estr else (str(repr(e)) + add_bt(e))
            new_e = errors.TypingError(textwrap.dedent(estr))
            return tmplt.format(literalness, str(new_e))

        # Bodo change
        import bodo

        if isinstance(literal_e, bodo.utils.typing.BodoError):
            raise literal_e
        # TODO: [BE-486] use environment variable
        if numba.core.config.DEVELOPER_MODE:
            raise errors.TypingError(
                nested_msg("literal", literal_e)
                + nested_msg("non-literal", nonliteral_e)
            )
        else:
            # Suppress numba stack trace and use our simplified error message
            # TODO: Disable Python traceback.
            # Message
            # a temporary solution.
            if "missing a required argument" in literal_e.msg:
                msg = "missing a required argument"
            else:
                msg = "Compilation error for "
                # TODO add other data types
                if isinstance(self.this, bodo.hiframes.pd_dataframe_ext.DataFrameType):
                    msg += "DataFrame."
                elif isinstance(self.this, bodo.hiframes.pd_series_ext.SeriesType):
                    msg += "Series."
                msg += f"{self.typing_key[1]}().{bodo_typing_error_info}"
            raise errors.TypingError(msg, loc=literal_e.loc)
    return out


if _check_numba_change:  # pragma: no cover
    # make sure get_call_type hasn't changed before replacing it
    lines = inspect.getsource(numba.core.types.functions.BoundFunction.get_call_type)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "502cd77c0084452e903a45a0f1f8107550bfbde7179363b57dabd617ce135f4a"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.types.functions.BoundFunction.get_call_type has changed"
        )


numba.core.types.functions.BoundFunction.get_call_type = get_call_type2


# ----------------------- unliteral monkey patch done ------------------------- #


# replace string_from_string_and_size since Numba 0.49 removes python 2.7 symbol support
# leading to a bug in this function
# https://github.com/numba/numba/blob/1ea770564cb3c0c6cb9d8ab92e7faf23cd4c4c19/numba/core/pythonapi.py#L1102
# TODO: remove when Numba is fixed
def string_from_string_and_size(self, string, size):
    from llvmlite import ir as lir

    fnty = lir.FunctionType(self.pyobj, [self.cstring, self.py_ssize_t])
    # replace PyString_FromStringAndSize with PyUnicode_FromStringAndSize of Python 3
    # fname = "PyString_FromStringAndSize"
    fname = "PyUnicode_FromStringAndSize"
    fn = self._get_function(fnty, name=fname)
    return self.builder.call(fn, [string, size])


numba.core.pythonapi.PythonAPI.string_from_string_and_size = string_from_string_and_size

# This replaces Numba's numba.core.dispatcher._DispatcherBase._compile_for_args
# method to delete args before returning the dispatcher object and handle BodoError.
# Otherwise, the code is the same.
def _compile_for_args(self, *args, **kws):  # pragma: no cover
    """
    For internal use.  Compile a specialized version of the function
    for the given *args* and *kws*, and return the resulting callable.
    """
    assert not kws
    # call any initialisation required for the compilation chain (e.g.
    # extension point registration).
    self._compilation_chain_init_hook()
    import bodo

    def error_rewrite(e, issue_type):
        """
        Rewrite and raise Exception `e` with help supplied based on the
        specified issue_type.
        """
        if numba.core.config.SHOW_HELP:
            help_msg = errors.error_extras[issue_type]
            e.patch_message("\n".join((str(e).rstrip(), help_msg)))
        if numba.core.config.FULL_TRACEBACKS:
            raise e
        else:
            raise e.with_traceback(None)

    argtypes = []
    for a in args:
        if isinstance(a, numba.core.dispatcher.OmittedArg):
            argtypes.append(types.Omitted(a.value))
        else:
            argtypes.append(self.typeof_pyval(a))
    return_val = None
    try:
        error = None
        return_val = self.compile(tuple(argtypes))
    except errors.ForceLiteralArg as e:
        # Received request for compiler re-entry with the list of arguments
        # indicated by e.requested_args.
        # First, check if any of these args are already Literal-ized
        # Bodo change:
        # do not consider LiteralStrKeyDict a literal since its values are not consts
        already_lit_pos = [
            i
            for i in e.requested_args
            if isinstance(args[i], types.Literal)
            and not isinstance(args[i], types.LiteralStrKeyDict)
        ]
        if already_lit_pos:
            # Abort compilation if any argument is already a Literal.
            # Letting this continue will cause infinite compilation loop.
            m = (
                "Repeated literal typing request.\n"
                "{}.\n"
                "This is likely caused by an error in typing. "
                "Please see nested and suppressed exceptions."
            )
            info = ", ".join(
                "Arg #{} is {}".format(i, args[i]) for i in sorted(already_lit_pos)
            )
            raise errors.CompilerError(m.format(info))
        # Convert requested arguments into a Literal.
        # Bodo change: requested args with FileInfo object are converted to FilenameType
        new_args = []
        try:
            for i, v in enumerate(args):
                if i in e.requested_args:
                    if i in e.file_infos:
                        new_args.append(types.FilenameType(args[i], e.file_infos[i]))
                    else:
                        new_args.append(types.literal(args[i]))
                else:
                    new_args.append(args[i])
            args = new_args
        # exception comes from find_file_name_or_handler in fs_io.py called by FilenameType
        # OSError: When AWS credentials are not provided/incorrect
        except (OSError, FileNotFoundError) as ferr:
            error = FileNotFoundError(str(ferr) + "\n" + e.loc.strformat() + "\n")
        # This is done to suppress stack when error comes as BodoError called by FilenameType.
        except bodo.utils.typing.BodoError as e:
            error = bodo.utils.typing.BodoError(str(e))
        # Re-enter compilation with the Literal-ized arguments
        # only if there's no problem with FilenameType
        if error is None:
            try:
                # This might raise TypingError/BodoError
                return_val = self._compile_for_args(*args)
            except TypingError as e:
                # Set error to be raised in finally section
                error = errors.TypingError(str(e))
            except bodo.utils.typing.BodoError as e:
                # Set error to be raised in finally section
                error = bodo.utils.typing.BodoError(str(e))

    except errors.TypingError as e:
        # Intercept typing error that may be due to an argument
        # that failed inferencing as a Numba type
        failed_args = []
        for i, arg in enumerate(args):
            val = (
                arg.value if isinstance(arg, numba.core.dispatcher.OmittedArg) else arg
            )
            try:
                tp = typeof(val, Purpose.argument)
            except ValueError as typeof_exc:
                failed_args.append((i, str(typeof_exc)))
            else:
                if tp is None:
                    failed_args.append(
                        (i, f"cannot determine Numba type of value {val}")
                    )
        if failed_args:
            # Patch error message to ease debugging
            args_str = "\n".join(f"- argument {i}: {err}" for i, err in failed_args)
            msg = (
                f"{str(e).rstrip()} \n\nThis error may have been caused "
                f"by the following argument(s):\n{args_str}\n"
            )
            e.patch_message(msg)

        if "Cannot determine Numba type of <class 'numpy.ufunc'>" in e.msg:
            # If we see an inability to type a known function, replace it with
            # a cleaner error message. Here we rely on e.loc to point us to
            # the function since we don't have enough information to extract
            # the function name.
            msg = "Unsupported Numpy ufunc encountered in JIT code"
            error = bodo.utils.typing.BodoError(msg, loc=e.loc)

        # In user mode if error comes from numba lowering, suppress stack.
        # Only if it has not been suppressed earlier (because of TypingError in Bodo).
        elif not numba.core.config.DEVELOPER_MODE:
            # If error_info is already there, that means Bodo already suppressed stack.
            if bodo_typing_error_info not in e.msg:
                # This is a Numba error
                numba_deny_list = [
                    "Failed in nopython mode pipeline",
                    "Failed in bodo mode pipeline",
                    "Failed at nopython",
                    "Overload",
                    "lowering",
                ]
                n_found = False
                for n_msg in numba_deny_list:
                    if n_msg in e.msg:
                        msg = "Compilation error. "
                        msg += f"{bodo_typing_error_info}"
                        n_found = True
                        break
                if not n_found:
                    msg = f"{str(e)}"
                msg += "\n" + e.loc.strformat() + "\n"
                e.patch_message(msg)
        error_rewrite(e, "typing")
    except errors.UnsupportedError as e:
        # Something unsupported is present in the user code, add help info
        error_rewrite(e, "unsupported_error")
    except (
        errors.NotDefinedError,
        errors.RedefinedError,
        errors.VerificationError,
    ) as e:
        # These errors are probably from an issue with either the code supplied
        # being syntactically or otherwise invalid
        error_rewrite(e, "interpreter")
    except errors.ConstantInferenceError as e:
        # this is from trying to infer something as constant when it isn't
        # or isn't supported as a constant
        error_rewrite(e, "constant_inference")
    # Bodo change: handle BodoError
    except bodo.utils.typing.BodoError as e:
        # create a new error so that the stacktrace only reaches
        # the point where the new error is raised
        error = bodo.utils.typing.BodoError(str(e))
    except Exception as e:
        if numba.core.config.SHOW_HELP:
            if hasattr(e, "patch_message"):
                help_msg = errors.error_extras["reportable"]
                e.patch_message("\n".join((str(e).rstrip(), help_msg)))
        # ignore the FULL_TRACEBACKS config, this needs reporting!
        raise e
    # Bodo change: avoid arg leak
    finally:
        self._types_active_call = []
        # avoid issue of reference leak of arguments to jitted function:
        # https://github.com/numba/numba/issues/5419
        del args
        if error:
            raise error
    return return_val


# workaround for Numba #5419 issue (https://github.com/numba/numba/issues/5419)
# first we check that the hash of the Numba function that we are replacing
# matches the one of the function that we copied from Numba

if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.dispatcher._DispatcherBase._compile_for_args)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "5cdfbf0b13a528abf9f0408e70f67207a03e81d610c26b1acab5b2dc1f79bf06"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.dispatcher._DispatcherBase._compile_for_args has changed"
        )

# now replace the function with our own
numba.core.dispatcher._DispatcherBase._compile_for_args = _compile_for_args


# TODO maybe we can do this in another function that we already monkey patch
# like _compile_for_args or our own decorator
def resolve_gb_agg_funcs(cres):
    from bodo.ir.aggregate import gb_agg_cfunc_addr

    # TODO? could there be a situation where we load multiple bodo functions
    # and name clashing occurs?
    for sym in cres.library._codegen._engine._defined_symbols:
        if (
            sym.startswith("cfunc")
            and ("get_agg_udf_addr" not in sym)
            and (
                "bodo_gb_udf_update_local" in sym
                or "bodo_gb_udf_combine" in sym
                or "bodo_gb_udf_eval" in sym
                or "bodo_gb_apply_general_udfs" in sym
            )
        ):
            gb_agg_cfunc_addr[sym] = cres.library.get_pointer_to_function(sym)


# TODO maybe we can do this in another function that we already monkey patch
# like _compile_for_args or our own decorator
def resolve_join_general_cond_funcs(cres):
    from bodo.ir.join import join_gen_cond_cfunc_addr

    # TODO? could there be a situation where we load multiple bodo functions
    # and name clashing occurs?
    for sym in cres.library._codegen._engine._defined_symbols:
        if sym.startswith("cfunc") and (
            "get_join_cond_addr" not in sym or "bodo_join_gen_cond" in sym
        ):
            join_gen_cond_cfunc_addr[sym] = cres.library.get_pointer_to_function(sym)


def compile(self, sig):
    import numba.core.event as ev
    from numba.core import sigutils
    from numba.core.compiler_lock import global_compiler_lock

    import bodo

    disp = self._get_dispatcher_for_current_target()
    if disp is not self:
        return disp.compile(sig)

    with ExitStack() as scope:
        cres = None

        def cb_compiler(dur):
            if cres is not None:
                self._callback_add_compiler_timer(dur, cres)

        def cb_llvm(dur):
            if cres is not None:
                self._callback_add_llvm_timer(dur, cres)

        scope.enter_context(ev.install_timer("numba:compiler_lock", cb_compiler))
        scope.enter_context(ev.install_timer("numba:llvm_lock", cb_llvm))
        scope.enter_context(global_compiler_lock)

        if not self._can_compile:
            raise RuntimeError("compilation disabled")
        # Use counter to track recursion compilation depth
        with self._compiling_counter:
            args, return_type = sigutils.normalize_signature(sig)
            # Don't recompile if signature already exists
            existing = self.overloads.get(tuple(args))
            if existing is not None:
                return existing.entry_point
            # Try to load from disk cache
            cres = self._cache.load_overload(sig, self.targetctx)
            if cres is not None:
                resolve_gb_agg_funcs(cres)  # Bodo change
                resolve_join_general_cond_funcs(cres)  # Bodo change
                self._cache_hits[sig] += 1
                # XXX fold this in add_overload()? (also see compiler.py)
                if not cres.objectmode:
                    self.targetctx.insert_user_function(
                        cres.entry_point, cres.fndesc, [cres.library]
                    )
                self.add_overload(cres)
                return cres.entry_point

            self._cache_misses[sig] += 1
            ev_details = dict(
                dispatcher=self,
                args=args,
                return_type=return_type,
            )
            with ev.trigger_event("numba:compile", data=ev_details):
                try:
                    cres = self._compiler.compile(args, return_type)
                except errors.ForceLiteralArg as e:

                    def folded(args, kws):
                        return self._compiler.fold_argument_types(args, kws)[1]

                    raise e.bind_fold_arguments(folded)
                self.add_overload(cres)
            # get_suitable_cache_subpath in numba.core.caching.py will perform a
            # hash of the parent directory path of the file being cached and append
            # this to the new cache folders path name, preventing name clashes
            if os.environ.get("BODO_PLATFORM_CACHE_LOCATION") is not None:
                # Since we used a shared file system on the platform, writing with just one rank is
                # sufficient, and desirable (to avoid I/O contention due to filesystem limitations).
                if bodo.get_rank() == 0:
                    self._cache.save_overload(sig, cres)
            else:
                # Even when not on platform, it's best to minimize I/O contention, so we
                # write cache files from one rank on each node.
                first_ranks = bodo.get_nodes_first_ranks()
                if bodo.get_rank() in first_ranks:
                    self._cache.save_overload(sig, cres)
            return cres.entry_point


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.dispatcher.Dispatcher.compile)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "934ec993577ea3b1c7dd2181ac02728abf8559fd42c17062cc821541b092ff8f"
    ):  # pragma: no cover
        warnings.warn("numba.core.dispatcher.Dispatcher.compile has changed")

numba.core.dispatcher.Dispatcher.compile = compile


def _get_module_for_linking(self):
    """
    Internal: get a LLVM module suitable for linking multiple times
    into another library.  Exported functions are made "linkonce_odr"
    to allow for multiple definitions, inlining, and removal of
    unused exports.

    See discussion in https://github.com/numba/numba/pull/890
    """
    import llvmlite.binding as ll  # Bodo change

    self._ensure_finalized()
    if self._shared_module is not None:
        return self._shared_module
    mod = self._final_module
    to_fix = []
    nfuncs = 0
    for fn in mod.functions:
        nfuncs += 1
        if not fn.is_declaration and fn.linkage == ll.Linkage.external:
            # Bodo change: skip groupby agg udf cfuncs, to avoid turning them
            # into weak symbols that are discarded
            if "get_agg_udf_addr" not in fn.name:
                if "bodo_gb_udf_update_local" in fn.name:
                    continue
                if "bodo_gb_udf_combine" in fn.name:
                    continue
                if "bodo_gb_udf_eval" in fn.name:
                    continue
                if "bodo_gb_apply_general_udfs" in fn.name:
                    continue
            # Bodo change: skip general join condition cfuncs, to avoid turning them
            # into weak symbols that are discarded
            if "get_join_cond_addr" not in fn.name:
                if "bodo_join_gen_cond" in fn.name:
                    continue
            to_fix.append(fn.name)
    if nfuncs == 0:
        # This is an issue which can occur if loading a module
        # from an object file and trying to link with it, so detect it
        # here to make debugging easier.
        raise RuntimeError(
            "library unfit for linking: " "no available functions in %s" % (self,)
        )
    if to_fix:
        mod = mod.clone()
        for name in to_fix:
            # NOTE: this will mark the symbol WEAK if serialized
            # to an ELF file
            mod.get_function(name).linkage = "linkonce_odr"
    self._shared_module = mod
    return mod


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.codegen.CPUCodeLibrary._get_module_for_linking)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "56dde0e0555b5ec85b93b97c81821bce60784515a1fbf99e4542e92d02ff0a73"
    ):
        warnings.warn(
            "numba.core.codegen.CPUCodeLibrary._get_module_for_linking has changed"
        )

numba.core.codegen.CPUCodeLibrary._get_module_for_linking = _get_module_for_linking


def propagate(self, typeinfer):
    """
    Execute all constraints.  Errors are caught and returned as a list.
    This allows progressing even though some constraints may fail
    due to lack of information
    (e.g. imprecise types such as List(undefined)).
    """
    import bodo

    errors = []
    for constraint in self.constraints:
        loc = constraint.loc
        with typeinfer.warnings.catch_warnings(filename=loc.filename, lineno=loc.line):
            try:
                constraint(typeinfer)
            except numba.core.errors.ForceLiteralArg as e:
                errors.append(e)
            except numba.core.errors.TypingError as e:
                numba.core.typeinfer._logger.debug("captured error", exc_info=e)
                new_exc = numba.core.errors.TypingError(
                    str(e),
                    loc=constraint.loc,
                    highlighting=False,
                )
                errors.append(numba.core.utils.chain_exception(new_exc, e))
            # Bodo change
            except bodo.utils.typing.BodoError as e:
                if loc not in e.locs_in_msg:
                    # the first time we see BodoError during type inference, we
                    # put the code location in the error message, and re-raise
                    errors.append(
                        bodo.utils.typing.BodoError(
                            str(e.msg) + "\n" + loc.strformat() + "\n",
                            locs_in_msg=e.locs_in_msg + [loc],
                        )
                    )
                else:
                    errors.append(
                        bodo.utils.typing.BodoError(e.msg, locs_in_msg=e.locs_in_msg)
                    )
            except Exception as e:
                from numba.core import utils

                if utils.use_old_style_errors():
                    numba.core.typeinfer._logger.debug("captured error", exc_info=e)
                    msg = (
                        "Internal error at {con}.\n{err}\n"
                        "Enable logging at debug level for details."
                    )
                    new_exc = numba.core.errors.TypingError(
                        msg.format(con=constraint, err=str(e)),
                        loc=constraint.loc,
                        highlighting=False,
                    )
                    errors.append(utils.chain_exception(new_exc, e))
                elif utils.use_new_style_errors():
                    raise e
                else:
                    msg = (
                        "Unknown CAPTURED_ERRORS style: "
                        f"'{numba.core.config.CAPTURED_ERRORS}'."
                    )
                    assert 0, msg
    return errors


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.typeinfer.ConstraintNetwork.propagate)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "1e73635eeba9ba43cb3372f395b747ae214ce73b729fb0adba0a55237a1cb063"
    ):
        warnings.warn("numba.core.typeinfer.ConstraintNetwork.propagate has changed")

numba.core.typeinfer.ConstraintNetwork.propagate = propagate


def raise_error(self):
    import bodo

    for faillist in self._failures.values():
        for fail in faillist:
            if isinstance(fail.error, ForceLiteralArg):
                raise fail.error
            # Bodo change
            if isinstance(fail.error, bodo.utils.typing.BodoError):
                raise fail.error
    raise TypingError(self.format())


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(
        numba.core.types.functions._ResolutionFailures.raise_error
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "84b89430f5c8b46cfc684804e6037f00a0f170005cd128ad245551787b2568ea"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.types.functions._ResolutionFailures.raise_error has changed"
        )

numba.core.types.functions._ResolutionFailures.raise_error = raise_error


# replaces remove_dead_block of Numba to add Bodo optimization (e.g. replace dead array
# in array.shape)
def bodo_remove_dead_block(
    block, lives, call_table, arg_aliases, alias_map, alias_set, func_ir, typemap
):
    """remove dead code using liveness info.
    Mutable arguments (e.g. arrays) that are not definitely assigned are live
    after return of function.
    """
    from bodo.transforms.distributed_pass import saved_array_analysis
    from bodo.utils.utils import is_array_typ, is_expr

    # TODO: find mutable args that are not definitely assigned instead of
    # assuming all args are live after return
    removed = False

    # add statements in reverse order
    new_body = [block.terminator]
    # for each statement in reverse order, excluding terminator
    for stmt in reversed(block.body[:-1]):
        # aliases of lives are also live
        alias_lives = set()
        init_alias_lives = lives & alias_set
        for v in init_alias_lives:
            alias_lives |= alias_map[v]
        lives_n_aliases = lives | alias_lives | arg_aliases

        # let external calls handle stmt if type matches
        if type(stmt) in remove_dead_extensions:
            f = remove_dead_extensions[type(stmt)]
            stmt = f(
                stmt, lives, lives_n_aliases, arg_aliases, alias_map, func_ir, typemap
            )
            if stmt is None:
                removed = True
                continue

        # ignore assignments that their lhs is not live or lhs==rhs
        if isinstance(stmt, ir.Assign):

            lhs = stmt.target
            rhs = stmt.value

            if lhs.name not in lives:
                if has_no_side_effect(rhs, lives_n_aliases, call_table):
                    removed = True
                    continue
                if (
                    isinstance(rhs, ir.Expr)
                    and rhs.op == "call"
                    and call_table[rhs.func.name] == ["astype"]
                ):
                    # Check if we can eliminate an np.array.astype call.  All other astypes should be handled within bodo, via inlining.
                    # This is needed for dead column elimination with BodoSQL
                    # This cannot be done within remove_hiframes, as we need access to the typing

                    fn_def = guard(get_definition, func_ir, rhs.func)
                    if (
                        fn_def is not None
                        and fn_def.op == "getattr"
                        and isinstance(typemap[fn_def.value.name], types.Array)
                        and fn_def.attr == "astype"
                    ):
                        removed = True
                        continue

            # replace dead array in array.shape with a live alternative equivalent array
            # this happens for CSV/Parquet read nodes where the first array is used
            # for forming RangeIndex but some other arrays may be used in the
            # program afterwards
            if (
                saved_array_analysis
                and lhs.name in lives
                and is_expr(rhs, "getattr")
                and rhs.attr == "shape"
                and is_array_typ(typemap[rhs.value.name])
                and rhs.value.name not in lives
            ):
                # TODO: use proper block to label mapping
                block_to_label = {v: k for k, v in func_ir.blocks.items()}
                # blocks inside parfors are not available in block_to_label
                # (see test_series_map_array_item_input without the isinstance check
                # above)
                if block in block_to_label:
                    label = block_to_label[block]
                    eq_set = saved_array_analysis.get_equiv_set(label)
                    var_eq_set = eq_set.get_equiv_set(rhs.value)
                    if var_eq_set is not None:
                        for v in var_eq_set:
                            if v.endswith("#0"):
                                v = v[:-2]
                            if v in typemap and is_array_typ(typemap[v]) and v in lives:
                                rhs.value = ir.Var(rhs.value.scope, v, rhs.value.loc)
                                removed = True
                                break

            if isinstance(rhs, ir.Var) and lhs.name == rhs.name:
                removed = True
                continue
            # TODO: remove other nodes like SetItem etc.

        if isinstance(stmt, ir.Del):
            if stmt.value not in lives:
                removed = True
                continue

        if isinstance(stmt, ir.SetItem):
            name = stmt.target.name
            if name not in lives_n_aliases:
                continue

        if type(stmt) in analysis.ir_extension_usedefs:
            def_func = analysis.ir_extension_usedefs[type(stmt)]
            uses, defs = def_func(stmt)
            lives -= defs
            lives |= uses
        else:
            lives |= {v.name for v in stmt.list_vars()}
            if isinstance(stmt, ir.Assign):
                # bodo change:
                # target variable of assignment is not live anymore only if it is not
                # used in right hand side. e.g. A = -A
                rhs_vars = set()
                if isinstance(rhs, ir.Expr):
                    rhs_vars = {v.name for v in rhs.list_vars()}
                if lhs.name not in rhs_vars:
                    lives.remove(lhs.name)

        new_body.append(stmt)
    new_body.reverse()
    block.body = new_body
    return removed


ir_utils.remove_dead_block = bodo_remove_dead_block


# replacing 'set' constructor typing of Numba to support string type
# TODO: declare string_type (unicode_type) hashable in Numba and remove this code
@infer_global(set)
class SetBuiltin(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        if args:
            # set(iterable)
            (iterable,) = args
            if isinstance(iterable, types.IterableType):
                dtype = iterable.iterator_type.yield_type
                if (
                    isinstance(dtype, types.Hashable)
                    or dtype == numba.core.types.unicode_type
                ):
                    return signature(types.Set(dtype), iterable)
        else:
            # set()
            return signature(types.Set(types.undefined))


# replacing types.Set.__init__ to support string dtype
def Set__init__(self, dtype, reflected=False):
    assert (
        isinstance(dtype, (types.Hashable, types.Undefined))
        or dtype == numba.core.types.unicode_type
    )
    self.dtype = dtype
    self.reflected = reflected
    cls_name = "reflected set" if reflected else "set"
    name = "%s(%s)" % (cls_name, self.dtype)
    super(types.Set, self).__init__(name=name)


types.Set.__init__ = Set__init__


# XXX: adding lowerer for eq of strings due to limitation of Set
@lower_builtin(operator.eq, types.UnicodeType, types.UnicodeType)
def eq_str(context, builder, sig, args):
    func = numba.cpython.unicode.unicode_eq(*sig.args)
    return context.compile_internal(builder, func, sig, args)


# disable push_call_vars() since it is only useful for threading not used in Bodo and
# it's buggy. See "test_series_combine"[S10-S20-None-False]"
numba.parfors.parfor.push_call_vars = (
    lambda blocks, saved_globals, saved_getattrs, typemap, nested=False: None
)


# replace Numba's maybe_literal to avoid using our ListLiteral in type inference
def maybe_literal(value):
    """Get a Literal type for the value or None."""
    # bodo change: don't use our ListLiteral for regular constant or global lists.
    # ListLiteral is only used when Bodo forces an argument to be a literal
    # FunctionLiteral shouldn't be used for all globals to avoid interference with
    # overloads
    if isinstance(value, (list, dict, pytypes.FunctionType)):
        return
    # Bodo change: support tuples of literal values (typeof_global handles tuples but
    # others like typeof_const go through resolve_value_type that calls maybe_literal)
    # see test_pd_categorical_compile_time
    if isinstance(value, tuple):
        try:
            return types.Tuple([literal(x) for x in value])
        except LiteralTypingError:
            return
    try:
        return literal(value)
    except LiteralTypingError:
        return


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(types.maybe_literal)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "8fb2fd93acf214b28e33e37d19dc2f7290a42792ec59b650553ac278854b5081"
    ):  # pragma: no cover
        warnings.warn("types.maybe_literal has changed")

types.maybe_literal = maybe_literal
types.misc.maybe_literal = maybe_literal


def CacheImpl__init__(self, py_func):
    self._lineno = py_func.__code__.co_firstlineno
    # Get qualname
    try:
        qualname = py_func.__qualname__
    except AttributeError:  # pragma: no cover
        qualname = py_func.__name__
    # Find a locator
    source_path = inspect.getfile(py_func)
    for cls in self._locator_classes:
        locator = cls.from_function(py_func, source_path)
        if locator is not None:
            break
    else:  # pragma: no cover
        raise RuntimeError(
            "cannot cache function %r: no locator available "
            "for file %r" % (qualname, source_path)
        )
    self._locator = locator
    # Use filename base name as module name to avoid conflict between
    # foo/__init__.py and foo/foo.py
    filename = inspect.getfile(py_func)
    modname = os.path.splitext(os.path.basename(filename))[0]

    # bodo change: correct the ipython module name by removing the cell number,
    # to guarantee that the cache file is found for the same function
    if source_path.startswith("<ipython-"):  # pragma: no cover
        new_modname = re.sub(
            r"(ipython-input)(-\d+)(-[0-9a-fA-F]+)", r"\1\3", modname, count=1
        )
        if new_modname == modname:
            warnings.warn(
                "Did not recognize ipython module name syntax. Caching might not work"
            )
        modname = new_modname

    fullname = "%s.%s" % (modname, qualname)
    abiflags = getattr(sys, "abiflags", "")
    self._filename_base = self.get_filename_base(fullname, abiflags)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.caching._CacheImpl.__init__)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "b46d298146e3844e9eaeef29d36f5165ba4796c270ca50d2b35f9fcdc0fa032a"
    ):  # pragma: no cover
        warnings.warn("numba.core.caching._CacheImpl.__init__ has changed")

numba.core.caching._CacheImpl.__init__ = CacheImpl__init__


# replacing _analyze_broadcast in array analysis to fix a bug. It's assuming that
# get_shape throws GuardException which is wrong.
# Numba 0.48 exposed this error with test_linear_regression since array analysis is
# more restrictive and assumes more variables as redefined
def _analyze_broadcast(self, scope, equiv_set, loc, args, fn):
    """Infer shape equivalence of arguments based on Numpy broadcast rules
    and return shape of output
    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    """
    from numba.parfors.array_analysis import ArrayAnalysis

    tups = list(filter(lambda a: self._istuple(a.name), args))
    # Here we have a tuple concatenation.
    if len(tups) == 2 and fn.__name__ == "add":
        # If either of the tuples is empty then the resulting shape
        # is just the other tuple.
        tup0typ = self.typemap[tups[0].name]
        tup1typ = self.typemap[tups[1].name]
        if tup0typ.count == 0:
            return ArrayAnalysis.AnalyzeResult(shape=equiv_set.get_shape(tups[1]))
        if tup1typ.count == 0:
            return ArrayAnalysis.AnalyzeResult(shape=equiv_set.get_shape(tups[0]))

        try:
            shapes = [equiv_set.get_shape(x) for x in tups]
            if None in shapes:
                return None
            concat_shapes = sum(shapes, ())
            return ArrayAnalysis.AnalyzeResult(shape=concat_shapes)
        except GuardException:
            return None

    # else arrays
    arrs = list(filter(lambda a: self._isarray(a.name), args))
    require(len(arrs) > 0)
    names = [x.name for x in arrs]
    dims = [self.typemap[x.name].ndim for x in arrs]
    max_dim = max(dims)
    require(max_dim > 0)
    # Bodo change:
    # try:
    #     shapes = [equiv_set.get_shape(x) for x in arrs]
    # except GuardException:
    #     return ArrayAnalysis.AnalyzeResult(
    #         shape=arrs[0],
    #         pre=self._call_assert_equiv(scope, loc, equiv_set, arrs)
    #     )
    # if None not in shapes:
    #     return self._broadcast_assert_shapes(
    #         scope, equiv_set, loc, shapes, names
    #     )
    # else:
    #     return self._insert_runtime_broadcast_call(
    #         scope, loc, arrs, max_dim
    #     )

    shapes = [equiv_set.get_shape(x) for x in arrs]
    if any(a is None for a in shapes):
        return ArrayAnalysis.AnalyzeResult(
            shape=arrs[0], pre=self._call_assert_equiv(scope, loc, equiv_set, arrs)
        )
    return self._broadcast_assert_shapes(scope, equiv_set, loc, shapes, names)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(
        numba.parfors.array_analysis.ArrayAnalysis._analyze_broadcast
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "6c91fec038f56111338ea2b08f5f0e7f61ebdab1c81fb811fe26658cc354e40f"
    ):  # pragma: no cover
        warnings.warn(
            "numba.parfors.array_analysis.ArrayAnalysis._analyze_broadcast has changed"
        )

numba.parfors.array_analysis.ArrayAnalysis._analyze_broadcast = _analyze_broadcast


def slice_size(self, index, dsize, equiv_set, scope, stmts):
    return None, None


# avoid slice analysis of Numba since it generates slice size variables but forgets the
# equivalence in subsequent runs and generates new variables.
# Disabling it may disallow parfor fusion for sliced arrays but this isn't common for
# Bodo workloads.
# See https://bodo.atlassian.net/browse/BE-2230
# TODO: fix slice analysis in Numba
numba.parfors.array_analysis.ArrayAnalysis.slice_size = slice_size


# support handling nested UDFs inside and outside the jit functions
def convert_code_obj_to_function(code_obj, caller_ir):
    """
    Converts a code object from a `make_function.code` attr in the IR into a
    python function, caller_ir is the FunctionIR of the caller and is used for
    the resolution of freevars.
    """
    import bodo

    fcode = code_obj.code
    nfree = len(fcode.co_freevars)

    # bodo change: support closures that have global/freevar (as well as literal)
    free_var_names = fcode.co_freevars
    if code_obj.closure is not None:
        # code_obj.closure is a tuple variable of freevar variables
        assert isinstance(code_obj.closure, ir.Var)
        items, op = ir_utils.find_build_sequence(caller_ir, code_obj.closure)
        assert op == "build_tuple"
        free_var_names = [v.name for v in items]

    # bodo change: brought glbls upfront to be able to update with function globals
    # globals are the same as those in the caller.
    glbls = caller_ir.func_id.func.__globals__
    # UDF globals may be available (set in untyped pass), needed for BodoSQL (CASE UDFs)
    # Numba infrastructure returns a KeyError even if getattr has a default value.
    try:
        glbls = getattr(code_obj, "globals", glbls)
    except KeyError:
        pass

    # try and resolve freevars if they are consts in the caller's IR
    # these can be baked into the new function
    # Bodo change: new error message
    msg = (
        "Inner function is using non-constant variable '{}' from outer function. "
        "Please pass as argument if possible. See "
        "https://docs.bodo.ai/latest/api_docs/udfs/."
    )
    freevars = []
    for x in free_var_names:
        # not using guard here to differentiate between multiple definition and
        # non-const variable
        try:
            freevar_def = caller_ir.get_definition(x)
        except KeyError:
            raise bodo.utils.typing.BodoError(msg.format(x), loc=code_obj.loc)
        # bodo change: support Global/FreeVar and function constants/strs
        from numba.core.registry import CPUDispatcher

        if isinstance(freevar_def, (ir.Const, ir.Global, ir.FreeVar)):
            val = freevar_def.value
            if isinstance(val, str):
                val = "'{}'".format(val)
            # value can be constant function
            if isinstance(val, pytypes.FunctionType):
                func_name = ir_utils.mk_unique_var("nested_func").replace(".", "_")
                glbls[func_name] = bodo.jit(distributed=False)(val)
                # add a flag indicating that the Dispatcher is a converted nested func
                # that may need to be reverted back to regular Python in objmode.
                # done in untyped pass, see test_heterogeneous_series_box
                glbls[func_name].is_nested_func = True
                val = func_name
            if isinstance(val, CPUDispatcher):
                func_name = ir_utils.mk_unique_var("nested_func").replace(".", "_")
                glbls[func_name] = val
                val = func_name
            freevars.append(val)
        # bodo change: support nested lambdas using recursive call
        elif isinstance(freevar_def, ir.Expr) and freevar_def.op == "make_function":
            nested_func = convert_code_obj_to_function(freevar_def, caller_ir)
            func_name = ir_utils.mk_unique_var("nested_func").replace(".", "_")
            glbls[func_name] = bodo.jit(distributed=False)(nested_func)
            glbls[func_name].is_nested_func = True
            freevars.append(func_name)
        else:
            raise bodo.utils.typing.BodoError(msg.format(x), loc=code_obj.loc)

    func_env = "\n".join(["\tc_%d = %s" % (i, x) for i, x in enumerate(freevars)])
    func_clo = ",".join(["c_%d" % i for i in range(nfree)])
    co_varnames = list(fcode.co_varnames)

    # This is horrible. The code object knows about the number of args present
    # it also knows the name of the args but these are bundled in with other
    # vars in `co_varnames`. The make_function IR node knows what the defaults
    # are, they are defined in the IR as consts. The following finds the total
    # number of args (args + kwargs with defaults), finds the default values
    # and infers the number of "kwargs with defaults" from this and then infers
    # the number of actual arguments from that.
    n_kwargs = 0
    n_allargs = fcode.co_argcount
    kwarg_defaults = caller_ir.get_definition(code_obj.defaults)
    if kwarg_defaults is not None:
        if isinstance(kwarg_defaults, tuple):
            d = [caller_ir.get_definition(x).value for x in kwarg_defaults]
            kwarg_defaults_tup = tuple(d)
        else:
            d = [caller_ir.get_definition(x).value for x in kwarg_defaults.items]
            kwarg_defaults_tup = tuple(d)
        n_kwargs = len(kwarg_defaults_tup)
    nargs = n_allargs - n_kwargs

    func_arg = ",".join(["%s" % (co_varnames[i]) for i in range(nargs)])
    if n_kwargs:
        kw_const = [
            "%s = %s" % (co_varnames[i + nargs], kwarg_defaults_tup[i])
            for i in range(n_kwargs)
        ]
        func_arg += ", "
        func_arg += ", ".join(kw_const)

    # create the function and return it
    return _create_function_from_code_obj(fcode, func_env, func_arg, func_clo, glbls)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.ir_utils.convert_code_obj_to_function)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "b840769812418d589460e924a15477e83e7919aac8a3dcb0188ff447344aa8ac"
    ):  # pragma: no cover
        warnings.warn("numba.core.ir_utils.convert_code_obj_to_function has changed")

numba.core.ir_utils.convert_code_obj_to_function = convert_code_obj_to_function
numba.core.untyped_passes.convert_code_obj_to_function = convert_code_obj_to_function


def passmanager_run(self, state):
    """
    Run the defined pipelines on the state.
    """
    from numba.core.compiler import _EarlyPipelineCompletion

    if not self.finalized:
        raise RuntimeError("Cannot run non-finalised pipeline")

    # Bodo change
    from numba.core.compiler_machinery import CompilerPass, _pass_registry

    import bodo

    # walk the passes and run them
    for idx, (pss, pass_desc) in enumerate(self.passes):
        try:
            numba.core.tracing.event("-- %s" % pass_desc)
            pass_inst = _pass_registry.get(pss).pass_inst
            if isinstance(pass_inst, CompilerPass):
                self._runPass(idx, pass_inst, state)
            else:
                raise BaseException("Legacy pass in use")
        except _EarlyPipelineCompletion as e:
            raise e
        # Bodo change
        except bodo.utils.typing.BodoError as e:
            raise
        except Exception as e:
            # TODO: [BE-486] environment variable developer_mode?
            if numba.core.config.DEVELOPER_MODE:
                from numba.core import utils

                if utils.use_new_style_errors() and not isinstance(
                    e, errors.NumbaError
                ):
                    raise e
                msg = "Failed in %s mode pipeline (step: %s)" % (
                    self.pipeline_name,
                    pass_desc,
                )
                patched_exception = self._patch_error(msg, e)
                raise patched_exception
            else:
                # Remove `Failed in ... pipeline` message
                raise e


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.compiler_machinery.PassManager.run)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "43505782e15e690fd2d7e53ea716543bec37aa0633502956864edf649e790cdb"
    ):  # pragma: no cover
        warnings.warn("numba.core.compiler_machinery.PassManager.run has changed")

numba.core.compiler_machinery.PassManager.run = passmanager_run


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.np.ufunc.parallel._launch_threads)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "a57ef28c4168fdd436a5513bba4351ebc6d9fba76c5819f44046431a79b9030f"
    ):  # pragma: no cover
        warnings.warn("numba.np.ufunc.parallel._launch_threads has changed")


# avoid launching threads in Numba, which may throw "omp_set_nested routine deprecated"
numba.np.ufunc.parallel._launch_threads = lambda: None


def get_reduce_nodes(reduction_node, nodes, func_ir):
    """
    Get nodes that combine the reduction variable with a sentinel variable.
    Recognizes the first node that combines the reduction variable with another
    variable.
    """
    reduce_nodes = None
    defs = {}

    def lookup(var, already_seen, varonly=True):
        val = defs.get(var.name, None)
        if isinstance(val, ir.Var):
            # Bodo Change: Track variables that are already
            # seen to avoid infinite recursion.
            # For example:
            #       $x = $y
            #       $y = $x
            # can lead to infinite recursion
            if val.name in already_seen:
                return var
            already_seen.add(val.name)
            return lookup(val, already_seen, varonly)
        else:
            return var if (varonly or val is None) else val

    name = reduction_node.name
    unversioned_name = reduction_node.unversioned_name
    for i, stmt in enumerate(nodes):
        lhs = stmt.target
        rhs = stmt.value
        defs[lhs.name] = rhs
        if isinstance(rhs, ir.Var) and rhs.name in defs:
            rhs = lookup(rhs, set())
        if isinstance(rhs, ir.Expr):
            in_vars = set(lookup(v, set(), True).name for v in rhs.list_vars())
            if name in in_vars:
                # Bodo change: avoid raising error for concat reduction case
                # opened issue to handle Bodo cases and raise proper errors: #1414
                # see test_concat_reduction

                # reductions like sum have an assignment afterwards
                # e.g. $2 = a + $1; a = $2
                # reductions that are functions calls like max() don't have an
                # extra assignment afterwards
                # if (not (i+1 < len(nodes) and isinstance(nodes[i+1], ir.Assign)
                #         and nodes[i+1].target.unversioned_name == unversioned_name)
                #         and lhs.unversioned_name != unversioned_name):
                #     raise ValueError(
                #         f"Use of reduction variable {unversioned_name!r} other "
                #         "than in a supported reduction function is not "
                #         "permitted."
                #     )

                # if not supported_reduction(rhs, func_ir):
                #     raise ValueError(("Use of reduction variable " + unversioned_name +
                #                       " in an unsupported reduction function."))
                args = [(x.name, lookup(x, set(), True)) for x in get_expr_args(rhs)]
                non_red_args = [x for (x, y) in args if y.name != name]
                # Bodo change: avoid raising error for concat reduction case
                # assert len(non_red_args) == 1
                args = [(x, y) for (x, y) in args if x != y.name]
                replace_dict = dict(args)
                # Bodo change: avoid error for concat reduction case
                if len(non_red_args) == 1:
                    replace_dict[non_red_args[0]] = ir.Var(
                        lhs.scope, name + "#init", lhs.loc
                    )
                replace_vars_inner(rhs, replace_dict)
                reduce_nodes = nodes[i:]
                break
    return reduce_nodes


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.parfors.parfor.get_reduce_nodes)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "a05b52aff9cb02e595a510cd34e973857303a71097fc5530567cb70ca183ef3b"
    ):  # pragma: no cover
        warnings.warn("numba.parfors.parfor.get_reduce_nodes has changed")


numba.parfors.parfor.get_reduce_nodes = get_reduce_nodes


# declare array writes in Bodo IR nodes and builtins to avoid invalid statement
# reordering in parfor fusion
def _can_reorder_stmts(stmt, next_stmt, func_ir, call_table, alias_map, arg_aliases):
    """
    Check dependencies to determine if a parfor can be reordered in the IR block
    with a non-parfor statement.
    """
    from numba.parfors.parfor import Parfor, expand_aliases, is_assert_equiv

    # swap only parfors with non-parfors
    # don't reorder calls with side effects (e.g. file close)
    # only read-read dependencies are OK
    # make sure there is no write-write, write-read dependencies
    if (
        isinstance(stmt, Parfor)
        and not isinstance(next_stmt, Parfor)
        and not isinstance(next_stmt, ir.Print)
        and (
            not isinstance(next_stmt, ir.Assign)
            or has_no_side_effect(next_stmt.value, set(), call_table)
            or guard(is_assert_equiv, func_ir, next_stmt.value)
        )
    ):
        stmt_accesses = expand_aliases(
            {v.name for v in stmt.list_vars()}, alias_map, arg_aliases
        )
        # Bodo change: add func_ir input
        stmt_writes = expand_aliases(
            get_parfor_writes(stmt, func_ir), alias_map, arg_aliases
        )
        next_accesses = expand_aliases(
            {v.name for v in next_stmt.list_vars()}, alias_map, arg_aliases
        )
        # Bodo change: add func_ir input
        next_writes = expand_aliases(
            get_stmt_writes(next_stmt, func_ir), alias_map, arg_aliases
        )
        if len((stmt_writes & next_accesses) | (next_writes & stmt_accesses)) == 0:
            return True
    return False


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.parfors.parfor._can_reorder_stmts)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "18caa9a01b21ab92b4f79f164cfdbc8574f15ea29deedf7bafdf9b0e755d777c"
    ):  # pragma: no cover
        warnings.warn("numba.parfors.parfor._can_reorder_stmts has changed")

numba.parfors.parfor._can_reorder_stmts = _can_reorder_stmts


# Bodo change: add func_ir input
def get_parfor_writes(parfor, func_ir):
    from numba.parfors.parfor import Parfor

    assert isinstance(parfor, Parfor)
    writes = set()
    blocks = parfor.loop_body.copy()
    blocks[-1] = parfor.init_block
    for block in blocks.values():
        for stmt in block.body:
            # Bodo change: add func_ir input
            writes.update(get_stmt_writes(stmt, func_ir))
            if isinstance(stmt, Parfor):
                # Bodo change: add func_ir input
                writes.update(get_parfor_writes(stmt, func_ir))
    return writes


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.parfors.parfor.get_parfor_writes)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "a7b29cd76832b6f6f1f2d2397ec0678c1409b57a6eab588bffd344b775b1546f"
    ):  # pragma: no cover
        warnings.warn("numba.parfors.parfor.get_parfor_writes has changed")

# only used locally here, no need to replace in Numba

# Bodo change: add func_ir input
def get_stmt_writes(stmt, func_ir):
    import bodo
    from bodo.utils.utils import is_call_assign

    # TODO: test bodo nodes
    writes = set()
    if isinstance(stmt, (ir.Assign, ir.SetItem, ir.StaticSetItem)):
        writes.add(stmt.target.name)
    # Bodo change: add Bodo nodes and builtins
    if isinstance(stmt, bodo.ir.aggregate.Aggregate):
        writes = {v.name for v in stmt.df_out_vars.values()}
        if stmt.out_key_vars is not None:
            writes.update({v.name for v in stmt.out_key_vars})
    if isinstance(stmt, (bodo.ir.csv_ext.CsvReader, bodo.ir.parquet_ext.ParquetReader)):
        writes = {v.name for v in stmt.out_vars}
    if isinstance(stmt, bodo.ir.join.Join):
        writes = {v.name for v in stmt.out_data_vars.values()}
    if isinstance(stmt, bodo.ir.sort.Sort):
        if not stmt.inplace:
            writes.update(
                {
                    v.name
                    for i, v in enumerate(stmt.out_vars)
                    if (i not in stmt.dead_var_inds and i not in stmt.dead_key_var_inds)
                }
            )
    if is_call_assign(stmt):
        fdef = guard(find_callname, func_ir, stmt.value)
        if fdef in (
            ("setitem_str_arr_ptr", "bodo.libs.str_arr_ext"),
            ("setna", "bodo.libs.array_kernels"),
            ("str_arr_item_to_numeric", "bodo.libs.str_arr_ext"),
            (
                "str_arr_setitem_int_to_str",
                "bodo.libs.str_arr_ext",
            ),
            ("str_arr_setitem_NA_str", "bodo.libs.str_arr_ext"),
            ("str_arr_set_not_na", "bodo.libs.str_arr_ext"),
            ("get_str_arr_item_copy", "bodo.libs.str_arr_ext"),
            ("set_bit_to_arr", "bodo.libs.int_arr_ext"),
        ):
            writes.add(stmt.value.args[0].name)
        if fdef == ("generate_table_nbytes", "bodo.utils.table_utils"):
            writes.add(stmt.value.args[1].name)
    return writes


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.ir_utils.get_stmt_writes)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "1a7a80b64c9a0eb27e99dc8eaae187bde379d4da0b74c84fbf87296d87939974"
    ):  # pragma: no cover
        warnings.warn("numba.core.ir_utils.get_stmt_writes has changed")

# only used locally here, no need to replace in Numba


def patch_message(self, new_message):
    """
    Change the error message to the given new message.
    """
    # Bodo change: Bodo needs access to updated message (which is different
    # to str(exception) which could also include source code location) in
    # some cases like bodo/utils/typing.py::get_udf_error_msg
    self.msg = new_message
    self.args = (new_message,) + self.args[1:]


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.errors.NumbaError.patch_message)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "ed189a428a7305837e76573596d767b6e840e99f75c05af6941192e0214fa899"
    ):  # pragma: no cover
        warnings.warn("numba.core.errors.NumbaError.patch_message has changed")

numba.core.errors.NumbaError.patch_message = patch_message

# --------------------- add_context ------------------------------
def add_context(self, msg):
    """
    Add contextual info.  The exception message is expanded with the new
    contextual information.
    Bodo: avoid adding During resolve call message.
    """
    # TODO:  [BE-486] development_mode environment variable?
    if numba.core.config.DEVELOPER_MODE:
        self.contexts.append(msg)
        f = _termcolor.errmsg("{0}") + _termcolor.filename("During: {1}")
        newmsg = f.format(self, msg)
        self.args = (newmsg,)
    else:
        # Bodo change: remove `During resolve call` message
        f = _termcolor.errmsg("{0}")
        newmsg = f.format(self)
        self.args = (newmsg,)
    return self


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.errors.NumbaError.add_context)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "6a388d87788f8432c2152ac55ca9acaa94dbc3b55be973b2cf22dd4ee7179ab8"
    ):  # pragma: no cover
        warnings.warn("numba.core.errors.NumbaError.add_context has changed")

numba.core.errors.NumbaError.add_context = add_context

# --------------------- jitclass support --------------------------


def _get_dist_spec_from_options(spec, **options):
    """get distribution spec for jitclass from options passed to @bodo.jitclass"""
    from bodo.transforms.distributed_analysis import Distribution

    dist_spec = {}

    if "distributed" in options:
        for field in options["distributed"]:
            dist_spec[field] = Distribution.OneD_Var

    if "distributed_block" in options:
        for field in options["distributed_block"]:
            dist_spec[field] = Distribution.OneD

    return dist_spec


# Bodo change: extra **options arg
def register_class_type(cls, spec, class_ctor, builder, **options):
    """
    Internal function to create a jitclass.

    Args
    ----
    cls: the original class object (used as the prototype)
    spec: the structural specification contains the field types.
    class_ctor: the numba type to represent the jitclass
    builder: the internal jitclass builder
    """
    import typing as pt

    from numba.core.typing.asnumbatype import as_numba_type

    import bodo

    # Bodo change: get distribution spec
    dist_spec = _get_dist_spec_from_options(spec, **options)
    returns_maybe_distributed = options.get("returns_maybe_distributed", True)

    # Normalize spec
    if spec is None:
        spec = OrderedDict()
    elif isinstance(spec, Sequence):
        spec = OrderedDict(spec)

    # Extend spec with class annotations.
    for attr, py_type in pt.get_type_hints(cls).items():
        if attr not in spec:
            spec[attr] = as_numba_type(py_type)

    jitclass_base._validate_spec(spec)

    # Fix up private attribute names
    spec = jitclass_base._fix_up_private_attr(cls.__name__, spec)

    # Copy methods from base classes
    clsdct = {}
    for basecls in reversed(inspect.getmro(cls)):
        clsdct.update(basecls.__dict__)

    methods, props, static_methods, others = {}, {}, {}, {}
    for k, v in clsdct.items():
        if isinstance(v, pytypes.FunctionType):
            methods[k] = v
        elif isinstance(v, property):
            props[k] = v
        elif isinstance(v, staticmethod):
            static_methods[k] = v
        else:
            others[k] = v

    # Check for name shadowing
    shadowed = (set(methods) | set(props) | set(static_methods)) & set(spec)
    if shadowed:
        raise NameError("name shadowing: {0}".format(", ".join(shadowed)))

    docstring = others.pop("__doc__", "")
    jitclass_base._drop_ignored_attrs(others)
    if others:
        msg = "class members are not yet supported: {0}"
        members = ", ".join(others.keys())
        raise TypeError(msg.format(members))

    for k, v in props.items():
        if v.fdel is not None:
            raise TypeError("deleter is not supported: {0}".format(k))

    # Bodo change: replace njit with bodo.jit
    jit_methods = {
        k: bodo.jit(returns_maybe_distributed=returns_maybe_distributed)(v)
        for k, v in methods.items()
    }

    jit_props = {}
    for k, v in props.items():
        dct = {}
        if v.fget:
            # Bodo change: replace njit with bodo.jit
            dct["get"] = bodo.jit(v.fget)
        if v.fset:
            # Bodo change: replace njit with bodo.jit
            dct["set"] = bodo.jit(v.fset)
        jit_props[k] = dct

    # Bodo change: replace njit with bodo.jit
    jit_static_methods = {k: bodo.jit(v.__func__) for k, v in static_methods.items()}

    # Instantiate class type
    class_type = class_ctor(
        cls,
        jitclass_base.ConstructorTemplate,
        spec,
        jit_methods,
        jit_props,
        jit_static_methods,
        dist_spec,  # Bodo change: pass dist spec
    )

    jit_class_dct = dict(class_type=class_type, __doc__=docstring)
    jit_class_dct.update(jit_static_methods)
    cls = jitclass_base.JitClassType(cls.__name__, (cls,), jit_class_dct)

    # Register resolution of the class object
    typingctx = numba.core.registry.cpu_target.typing_context
    typingctx.insert_global(cls, class_type)

    # Register class
    targetctx = numba.core.registry.cpu_target.target_context
    builder(class_type, typingctx, targetctx).register()
    as_numba_type.register(cls, class_type.instance_type)

    return cls


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(jitclass_base.register_class_type)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "005e6e2e89a47f77a19ba86305565050d4dbc2412fc4717395adf2da348671a9"
    ):  # pragma: no cover
        warnings.warn("jitclass_base.register_class_type has changed")


jitclass_base.register_class_type = register_class_type


# Bodo change: extra dist_spec arg/attribute
def ClassType__init__(
    self,
    class_def,
    ctor_template_cls,
    struct,
    jit_methods,
    jit_props,
    jit_static_methods,
    dist_spec=None,
):
    if dist_spec is None:
        dist_spec = {}
    self.class_name = class_def.__name__
    self.class_doc = class_def.__doc__
    self._ctor_template_class = ctor_template_cls
    self.jit_methods = jit_methods
    self.jit_props = jit_props
    self.jit_static_methods = jit_static_methods
    self.struct = struct
    self.dist_spec = dist_spec
    fielddesc = ",".join("{0}:{1}".format(k, v) for k, v in struct.items())
    distdesc = ",".join("{0}:{1}".format(k, v) for k, v in dist_spec.items())
    name = "{0}.{1}#{2:x}<{3}><{4}>".format(
        self.name_prefix, self.class_name, id(self), fielddesc, distdesc
    )
    super(types.misc.ClassType, self).__init__(name)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(types.misc.ClassType.__init__)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "2b848ea82946c88f540e81f93ba95dfa7cd66045d944152a337fe2fc43451c30"
    ):  # pragma: no cover
        warnings.warn("types.misc.ClassType.__init__ has changed")

types.misc.ClassType.__init__ = ClassType__init__


# redefine jitclass decorator with our own register_class_type() and add '**options'
def jitclass(cls_or_spec=None, spec=None, **options):
    """
    A function for creating a jitclass.
    Can be used as a decorator or function.

    Different use cases will cause different arguments to be set.

    If specified, ``spec`` gives the types of class fields.
    It must be a dictionary or sequence.
    With a dictionary, use collections.OrderedDict for stable ordering.
    With a sequence, it must contain 2-tuples of (fieldname, fieldtype).

    Any class annotations for field names not listed in spec will be added.
    For class annotation `x: T` we will append ``("x", as_numba_type(T))`` to
    the spec if ``x`` is not already a key in spec.


    Examples
    --------

    1) ``cls_or_spec = None``, ``spec = None``

    >>> @jitclass()
    ... class Foo:
    ...     ...

    2) ``cls_or_spec = None``, ``spec = spec``

    >>> @jitclass(spec=spec)
    ... class Foo:
    ...     ...

    3) ``cls_or_spec = Foo``, ``spec = None``

    >>> @jitclass
    ... class Foo:
    ...     ...

    4) ``cls_or_spec = spec``, ``spec = None``
    In this case we update ``cls_or_spec, spec = None, cls_or_spec``.

    >>> @jitclass(spec)
    ... class Foo:
    ...     ...

    5) ``cls_or_spec = Foo``, ``spec = spec``

    >>> JitFoo = jitclass(Foo, spec)

    Returns
    -------
    If used as a decorator, returns a callable that takes a class object and
    returns a compiled version.
    If used as a function, returns the compiled class (an instance of
    ``JitClassType``).
    """

    if cls_or_spec is not None and spec is None and not isinstance(cls_or_spec, type):
        # Used like
        # @jitclass([("x", intp)])
        # class Foo:
        #     ...
        spec = cls_or_spec
        cls_or_spec = None

    def wrap(cls):
        if numba.core.config.DISABLE_JIT:
            return cls
        else:
            from numba.experimental.jitclass.base import ClassBuilder

            return register_class_type(
                cls, spec, types.ClassType, ClassBuilder, **options
            )

    if cls_or_spec is None:
        return wrap
    else:
        return wrap(cls_or_spec)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(jitclass_decorators.jitclass)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "265f1953ee5881d1a5d90238d3c932cd300732e41495657e65bf51e59f7f4af5"
    ):  # pragma: no cover
        warnings.warn("jitclass_decorators.jitclass has changed")


# -------------------- ForceLiteralArg --------------------


def CallConstraint_resolve(self, typeinfer, typevars, fnty):
    assert fnty
    context = typeinfer.context

    r = numba.core.typeinfer.fold_arg_vars(typevars, self.args, self.vararg, self.kws)
    if r is None:
        # Cannot resolve call type until all argument types are known
        return
    pos_args, kw_args = r

    # Check argument to be precise
    for a in itertools.chain(pos_args, kw_args.values()):
        # Forbids imprecise type except array of undefined dtype
        if not a.is_precise() and not isinstance(a, types.Array):
            return

    # Resolve call type
    if isinstance(fnty, types.TypeRef):
        # Unwrap TypeRef
        fnty = fnty.instance_type
    try:
        sig = typeinfer.resolve_call(fnty, pos_args, kw_args)
    except ForceLiteralArg as e:
        # Adjust for bound methods
        folding_args = (
            (fnty.this,) + tuple(self.args)
            if isinstance(fnty, types.BoundFunction)
            else self.args
        )
        folded = e.fold_arguments(folding_args, self.kws)
        requested = set()
        unsatisified = set()
        new_file_infos = {}  # Bodo change: propagate file_infos
        for idx in e.requested_args:
            maybe_arg = typeinfer.func_ir.get_definition(folded[idx])
            if isinstance(maybe_arg, ir.Arg):
                requested.add(maybe_arg.index)
                if maybe_arg.index in e.file_infos:
                    new_file_infos[maybe_arg.index] = e.file_infos[maybe_arg.index]
            else:  # pragma: no cover
                unsatisified.add(idx)
        if unsatisified:  # pragma: no cover
            raise TypingError("Cannot request literal type.", loc=self.loc)
        elif requested:
            # Bodo change: propagate file_infos
            raise ForceLiteralArg(requested, loc=self.loc, file_infos=new_file_infos)
    if sig is None:
        # Note: duplicated error checking.
        #       See types.BaseFunction.get_call_type
        # Arguments are invalid => explain why
        headtemp = "Invalid use of {0} with parameters ({1})"
        args = [str(a) for a in pos_args]
        args += ["%s=%s" % (k, v) for k, v in sorted(kw_args.items())]
        head = headtemp.format(fnty, ", ".join(map(str, args)))
        desc = context.explain_function_type(fnty)
        msg = "\n".join([head, desc])
        raise TypingError(msg)

    typeinfer.add_type(self.target, sig.return_type, loc=self.loc)

    # If the function is a bound function and its receiver type
    # was refined, propagate it.
    if (
        isinstance(fnty, types.BoundFunction)
        and sig.recvr is not None
        and sig.recvr != fnty.this
    ):
        refined_this = context.unify_pairs(sig.recvr, fnty.this)
        if (
            refined_this is None and fnty.this.is_precise() and sig.recvr.is_precise()
        ):  # pragma: no cover
            msg = "Cannot refine type {} to {}".format(
                sig.recvr,
                fnty.this,
            )
            raise TypingError(msg, loc=self.loc)
        if refined_this is not None and refined_this.is_precise():
            refined_fnty = fnty.copy(this=refined_this)
            typeinfer.propagate_refined_type(self.func, refined_fnty)

    # If the return type is imprecise but can be unified with the
    # target variable's inferred type, use the latter.
    # Useful for code such as::
    #    s = set()
    #    s.add(1)
    # (the set() call must be typed as int64(), not undefined())
    if not sig.return_type.is_precise():
        target = typevars[self.target]
        if target.defined:
            targetty = target.getone()
            if context.unify_pairs(targetty, sig.return_type) == targetty:
                sig = sig.replace(return_type=targetty)

    self.signature = sig
    self._add_refine_map(typeinfer, typevars, sig)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.typeinfer.CallConstraint.resolve)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "c78cd8ffc64b836a6a2ddf0362d481b52b9d380c5249920a87ff4da052ce081f"
    ):  # pragma: no cover
        warnings.warn("numba.core.typeinfer.CallConstraint.resolve has changed")

numba.core.typeinfer.CallConstraint.resolve = CallConstraint_resolve


def ForceLiteralArg__init__(
    self, arg_indices, fold_arguments=None, loc=None, file_infos=None
):
    """
    Parameters
    ----------
    arg_indices : Sequence[int]
        requested positions of the arguments.
    fold_arguments: callable
        A function ``(tuple, dict) -> tuple`` that binds and flattens
        the ``args`` and ``kwargs``.
    loc : numba.ir.Loc or None
    file_infos : A dict that maps arg index to FileInfo object if the
                 argument specified by that index must be converted to
                 FilenameType
    """
    super(ForceLiteralArg, self).__init__(
        "Pseudo-exception to force literal arguments in the dispatcher",
        loc=loc,
    )
    self.requested_args = frozenset(arg_indices)
    self.fold_arguments = fold_arguments
    # Bodo change: file info object to force FilenameType instead of Literal
    if file_infos is None:
        self.file_infos = {}
    else:
        self.file_infos = file_infos


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.errors.ForceLiteralArg.__init__)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "b241d5e36a4cf7f4c73a7ad3238693612926606c7a278cad1978070b82fb55ef"
    ):  # pragma: no cover
        warnings.warn("numba.core.errors.ForceLiteralArg.__init__ has changed")

numba.core.errors.ForceLiteralArg.__init__ = ForceLiteralArg__init__


def ForceLiteralArg_bind_fold_arguments(self, fold_arguments):
    """Bind the fold_arguments function"""
    # Bodo change: propagate file_infos
    e = ForceLiteralArg(
        self.requested_args, fold_arguments, loc=self.loc, file_infos=self.file_infos
    )
    return numba.core.utils.chain_exception(e, self)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.errors.ForceLiteralArg.bind_fold_arguments)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "1e93cca558f7c604a47214a8f2ec33ee994104cb3e5051166f16d7cc9315141d"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.errors.ForceLiteralArg.bind_fold_arguments has changed"
        )

numba.core.errors.ForceLiteralArg.bind_fold_arguments = (
    ForceLiteralArg_bind_fold_arguments
)


def ForceLiteralArg_combine(self, other):  # pragma: no cover
    """Returns a new instance by or'ing the requested_args."""
    if not isinstance(other, ForceLiteralArg):
        m = "*other* must be a {} but got a {} instead"
        raise TypeError(m.format(ForceLiteralArg, type(other)))
    # Bodo change: propagate file_infos
    return ForceLiteralArg(
        # for file infos, we merge the two dicts
        self.requested_args | other.requested_args,
        file_infos={**self.file_infos, **other.file_infos},
    )


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.errors.ForceLiteralArg.combine)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "49bf06612776f5d755c1c7d1c5eb91831a57665a8fed88b5651935f3bf33e899"
    ):  # pragma: no cover
        warnings.warn("numba.core.errors.ForceLiteralArg.combine has changed")

numba.core.errors.ForceLiteralArg.combine = ForceLiteralArg_combine


def _get_global_type(self, gv):
    from bodo.utils.typing import FunctionLiteral

    ty = self._lookup_global(gv)
    if ty is not None:
        return ty
    if isinstance(gv, pytypes.ModuleType):
        return types.Module(gv)

    # Bodo change: use FunctionLiteral for function value if it's not overloaded
    if isinstance(gv, pytypes.FunctionType):
        return FunctionLiteral(gv)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.typing.context.BaseContext._get_global_type)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "8ffe6b81175d1eecd62a37639b5005514b4477d88f35f5b5395041ac8c945a4a"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.typing.context.BaseContext._get_global_type has changed"
        )

numba.core.typing.context.BaseContext._get_global_type = _get_global_type


def _legalize_args(self, func_ir, args, kwargs, loc, func_globals, func_closures):
    """
    Legalize arguments to the context-manager

    Parameters
    ----------
    func_ir: FunctionIR
    args: tuple
        Positional arguments to the with-context call as IR nodes.
    kwargs: dict
        Keyword arguments to the with-context call as IR nodes.
    loc: numba.core.ir.Loc
        Source location of the with-context call.
    func_globals: dict
        The globals dictionary of the calling function.
    func_closures: dict
        The resolved closure variables of the calling function.
    """
    from numba.core import sigutils

    from bodo.utils.transform import get_const_value_inner

    if args:
        raise errors.CompilerError(
            "objectmode context doesn't take any positional arguments",
        )
    typeanns = {}

    def report_error(varname, msg, loc):
        raise errors.CompilerError(
            f"Error handling objmode argument {varname!r}. {msg}",
            loc=loc,
        )

    for k, v in kwargs.items():
        # Bodo change: use get_const_value_inner to find constant type value to support
        # more complex cases like bodo.int64[::1]
        v_const = None
        try:
            # create a dummy var to pass to get_const_value_inner since v is an IR node
            val_var = ir.Var(ir.Scope(None, loc), ir_utils.mk_unique_var("dummy"), loc)
            func_ir._definitions[val_var.name] = [v]
            v_const = get_const_value_inner(func_ir, val_var)
            func_ir._definitions.pop(val_var.name)
            if isinstance(v_const, str):
                v_const = sigutils._parse_signature_string(v_const)
            if isinstance(v_const, types.abstract._TypeMetaclass):
                # TODO(ehsan): add link to objmode docs when ready
                raise BodoError(
                    (
                        f"objmode type annotations require full data types, not just data type "
                        f"classes. For example, 'bodo.DataFrameType((bodo.float64[::1],), "
                        f"bodo.RangeIndexType(), ('A',))' is a valid data type but 'bodo.DataFrameType' is not.\n"
                        f"Variable {k} is annotated as type class {v_const}."
                    )
                )
            assert isinstance(v_const, types.Type)
            # list/set reflection is irrelevant in objmode
            if isinstance(v_const, (types.List, types.Set)):
                v_const = v_const.copy(reflected=False)
            typeanns[k] = v_const
        except BodoError:
            raise
        except:
            # recreate error messages similar to Numba
            msg = (
                "The value must be a compile-time constant either as "
                "a non-local variable or an expression that "
                "refers to a Bodo type."
            )
            if isinstance(v_const, ir.UndefinedType):
                msg = f"not defined."
                if isinstance(v, ir.Global):
                    msg = f"Global {v.name!r} is not defined."
                if isinstance(v, ir.FreeVar):
                    msg = f"Freevar {v.name!r} is not defined."

            if isinstance(v, ir.Expr) and v.op == "getattr":
                msg = "Getattr cannot be resolved at compile-time."
            report_error(
                varname=k,
                msg=msg,
                loc=loc,
            )

    # Legalize the types for objmode
    for name, typ in typeanns.items():
        self._legalize_arg_type(name, typ, loc)

    return typeanns


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(
        numba.core.withcontexts._ObjModeContextType._legalize_args
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "867c9ba7f1bcf438be56c38e26906bb551f59a99f853a9f68b71208b107c880e"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.withcontexts._ObjModeContextType._legalize_args has changed"
        )


numba.core.withcontexts._ObjModeContextType._legalize_args = _legalize_args


# Support for f-strings
# TODO(ehsan): remove when Numba's #6608 is merged
def op_FORMAT_VALUE_byteflow(self, state, inst):
    """
    FORMAT_VALUE(flags): flags argument specifies conversion (not supported yet) and
    format spec.
    Pops a value from stack and pushes results back.
    Required for supporting f-strings.
    https://docs.python.org/3/library/dis.html#opcode-FORMAT_VALUE
    """
    # check for conversion flags
    flags = inst.arg
    if (flags & 0x03) != 0x00:
        msg = "str/repr/ascii conversion in f-strings not supported yet"
        raise errors.UnsupportedError(msg, loc=self.get_debug_loc(inst.lineno))

    # if format specified
    format_spec = None
    if (flags & 0x04) == 0x04:
        format_spec = state.pop()

    value = state.pop()
    fmtvar = state.make_temp()
    res = state.make_temp()
    state.append(inst, value=value, res=res, fmtvar=fmtvar, format_spec=format_spec)
    state.push(res)


def op_BUILD_STRING_byteflow(self, state, inst):
    """
    BUILD_STRING(count): Concatenates count strings from the stack and pushes the
    resulting string onto the stack.
    Required for supporting f-strings.
    https://docs.python.org/3/library/dis.html#opcode-BUILD_STRING
    """
    count = inst.arg
    assert count > 0, "invalid BUILD_STRING count"
    strings = list(reversed([state.pop() for _ in range(count)]))
    tmps = [state.make_temp() for _ in range(count - 1)]
    state.append(inst, strings=strings, tmps=tmps)
    state.push(tmps[-1])


numba.core.byteflow.TraceRunner.op_FORMAT_VALUE = op_FORMAT_VALUE_byteflow
numba.core.byteflow.TraceRunner.op_BUILD_STRING = op_BUILD_STRING_byteflow


def op_FORMAT_VALUE_interpreter(self, inst, value, res, fmtvar, format_spec):
    """
    FORMAT_VALUE(flags): flags argument specifies conversion (not supported yet) and
    format spec.
    https://docs.python.org/3/library/dis.html#opcode-FORMAT_VALUE
    """
    value = self.get(value)
    fmtgv = ir.Global("format", format, loc=self.loc)
    self.store(value=fmtgv, name=fmtvar)
    args = (value, self.get(format_spec)) if format_spec else (value,)
    call = ir.Expr.call(self.get(fmtvar), args, (), loc=self.loc)
    self.store(value=call, name=res)


def op_BUILD_STRING_interpreter(self, inst, strings, tmps):
    """
    BUILD_STRING(count): Concatenates count strings.
    Required for supporting f-strings.
    https://docs.python.org/3/library/dis.html#opcode-BUILD_STRING
    """
    count = inst.arg
    assert count > 0, "invalid BUILD_STRING count"
    prev = self.get(strings[0])
    for other, tmp in zip(strings[1:], tmps):
        other = self.get(other)
        expr = ir.Expr.binop(operator.add, lhs=prev, rhs=other, loc=self.loc)
        self.store(expr, tmp)
        prev = self.get(tmp)


numba.core.interpreter.Interpreter.op_FORMAT_VALUE = op_FORMAT_VALUE_interpreter
numba.core.interpreter.Interpreter.op_BUILD_STRING = op_BUILD_STRING_interpreter


# add PyObject_HasAttrString call to pythonapi to be available in boxing/unboxing calls
# as c.pyapi.object_hasattr_string(), TODO(ehsan): move to Numba
def object_hasattr_string(self, obj, attr):
    from llvmlite import ir as lir

    cstr = self.context.insert_const_string(self.module, attr)
    fnty = lir.FunctionType(lir.IntType(32), [self.pyobj, self.cstring])
    fn = self._get_function(fnty, name="PyObject_HasAttrString")
    return self.builder.call(fn, [obj, cstr])


numba.core.pythonapi.PythonAPI.object_hasattr_string = object_hasattr_string


# BODO change: add mk_unique_var (see https://github.com/numba/numba/issues/7365)
def _created_inlined_var_name(function_name, var_name):
    """Creates a name for an inlined variable based on the function name and the
    variable name. It does this "safely" to avoid the use of characters that are
    illegal in python variable names as there are occasions when function
    generation needs valid python name tokens."""
    # Bodo change: add mk_unique_var() to make sure inlined variable names are unique
    # Bodo change: remove function_name to avoid very long variable names
    inlined_name = mk_unique_var(f"{var_name}")
    # Replace angle brackets, e.g. "<locals>" is replaced with "_locals_"
    new_name = inlined_name.replace("<", "_").replace(">", "_")
    # The version "version" of the closure function e.g. foo$2 (id 2) is
    # rewritten as "foo_v2". Further "." is also replaced with "_".
    new_name = new_name.replace(".", "_").replace("$", "_v")
    return new_name


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.inline_closurecall._created_inlined_var_name)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "0d91aac55cd0243e58809afe9d252562f9ae2899cde1112cc01a46804e01821e"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.inline_closurecall._created_inlined_var_name has changed"
        )

numba.core.inline_closurecall._created_inlined_var_name = _created_inlined_var_name


#################   Start Typer Changes   #################
# Numba 0.54 introduced more strict type checking for numpy type constructors.
# Since these are done outside of regular overloads, it is not possible to
# provide an additional overload that handles more general cases (since the
# type constructor cannot be extended). To get around this, we replace the typers
# in the functions that need additional constructors.

# TODO: Include directly in Numba

# Bodo Change, add support for calling a number constructor on strings, datetime, timedelta
def resolve_number___call__(self, classty):
    """
    Resolve a number class's constructor (e.g. calling int(...))
    """
    import numpy as np
    from numba.core.typing.templates import make_callable_template

    import bodo

    ty = classty.instance_type

    if isinstance(ty, types.NPDatetime):

        def typer(val1, val2):
            bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
                val1, "numpy.datetime64"
            )
            if val1 == bodo.hiframes.pd_timestamp_ext.pd_timestamp_type:
                if not is_overload_constant_str(val2):
                    raise_bodo_error(
                        "datetime64(): 'units' must be a 'str' specifying 'ns'"
                    )
                units = get_overload_const_str(val2)
                if units != "ns":
                    raise BodoError("datetime64(): 'units' must be 'ns'")
                return types.NPDatetime("ns")

    else:

        def typer(val):
            if isinstance(val, (types.BaseTuple, types.Sequence)):
                # Array constructor, e.g. np.int32([1, 2])
                fnty = self.context.resolve_value_type(np.array)
                sig = fnty.get_call_type(self.context, (val, types.DType(ty)), {})
                return sig.return_type
            elif isinstance(val, (types.Number, types.Boolean, types.IntEnumMember)):
                # Scalar constructor, e.g. np.int32(42)
                return ty
            # Bodo Change: Support for unicode
            elif val == types.unicode_type:
                return ty
            elif isinstance(val, (types.NPDatetime, types.NPTimedelta)):
                # Constructor cast from datetime-like, e.g.
                # > np.int64(np.datetime64("2000-01-01"))
                if ty.bitwidth == 64:
                    return ty
                else:
                    msg = f"Cannot cast {val} to {ty} as {ty} is not 64 bits " "wide."
                    raise errors.TypingError(msg)
            else:
                if isinstance(val, types.Array) and val.ndim == 0 and val.dtype == ty:
                    # This is 0d array -> scalar degrading
                    return ty
                else:
                    # unsupported
                    msg = f"Casting {val} to {ty} directly is unsupported."
                    if isinstance(val, types.Array):
                        # array casts are supported a different way.
                        msg += f" Try doing '<array>.astype(np.{ty})' instead"
                    raise errors.TypingError(msg)

    return types.Function(make_callable_template(key=ty, typer=typer))


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(
        numba.core.typing.builtins.NumberClassAttribute.resolve___call__
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "fdaf0c7d0820130481bb2bd922985257b9281b670f0bafffe10e51cabf0d5081"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.typing.builtins.NumberClassAttribute.resolve___call__ has changed"
        )

numba.core.typing.builtins.NumberClassAttribute.resolve___call__ = (
    resolve_number___call__
)


#################   End Typer Changes   #################


def on_assign(self, states, assign):

    if assign.target.name == states["varname"]:
        scope = states["scope"]
        defmap = states["defmap"]
        # Allow first assignment to retain the name
        if len(defmap) == 0:
            newtarget = assign.target
            numba.core.ssa._logger.debug("first assign: %s", newtarget)
            if newtarget.name not in scope.localvars:
                # Bodo Change: Define the variable if its not already in the scope.
                # and comment out warning so users don't see it.

                # wmsg = f"variable {newtarget.name!r} is not in scope."
                # warnings.warn(errors.NumbaIRAssumptionWarning(wmsg, loc=assign.loc))
                newtarget = scope.define(assign.target.name, loc=assign.loc)
        else:
            newtarget = scope.redefine(assign.target.name, loc=assign.loc)
        assign = ir.Assign(target=newtarget, value=assign.value, loc=assign.loc)
        defmap[states["label"]].append(assign)
    return assign


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.ssa._FreshVarHandler.on_assign)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "922c4f9807455f81600b794bbab36f9c6edfecfa83fda877bf85f465db7865e8"
    ):  # pragma: no cover
        warnings.warn("_FreshVarHandler on_assign has changed")

numba.core.ssa._FreshVarHandler.on_assign = on_assign


#################   Start Array Math Changes   #################

# Bodo change: Enable multiple overloads in canonicalize_array_math
def get_np_ufunc_typ_lst(func):
    """get type of the incoming function from builtin registry.
    Return all registered implementations"""
    from numba.core import typing

    impls = []
    for (k, v) in typing.npydecl.registry.globals:
        if k == func:
            impls.append(v)
    for (k, v) in typing.templates.builtin_registry.globals:
        if k == func:
            impls.append(v)
    if len(impls) == 0:
        raise RuntimeError("type for func ", func, " not found")
    return impls


def canonicalize_array_math(func_ir, typemap, calltypes, typingctx):
    import numpy
    from numba.core.ir_utils import arr_math, find_topo_order, mk_unique_var

    # save array arg to call
    # call_varname -> array
    blocks = func_ir.blocks
    saved_arr_arg = {}
    topo_order = find_topo_order(blocks)
    # Bodo Change: Store the implementations outside the typemap
    impl_map = {}
    for label in topo_order:
        block = blocks[label]
        new_body = []
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                lhs = stmt.target.name
                rhs = stmt.value
                # replace A.func with np.func, and save A in saved_arr_arg
                if (
                    rhs.op == "getattr"
                    and rhs.attr in arr_math
                    and isinstance(typemap[rhs.value.name], types.npytypes.Array)
                ):
                    rhs = stmt.value
                    arr = rhs.value
                    saved_arr_arg[lhs] = arr
                    scope = arr.scope
                    loc = arr.loc
                    # g_np_var = Global(numpy)
                    g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
                    typemap[g_np_var.name] = types.misc.Module(numpy)
                    g_np = ir.Global("np", numpy, loc)
                    g_np_assign = ir.Assign(g_np, g_np_var, loc)
                    rhs.value = g_np_var
                    new_body.append(g_np_assign)
                    func_ir._definitions[g_np_var.name] = [g_np]
                    # update func var type
                    func = getattr(numpy, rhs.attr)
                    # Bodo Change, don't store in typemap and select
                    # all possible implementations
                    func_typ_list = get_np_ufunc_typ_lst(func)
                    impl_map[lhs] = func_typ_list
                if rhs.op == "call" and rhs.func.name in saved_arr_arg:
                    # add array as first arg
                    arr = saved_arr_arg[rhs.func.name]
                    # update call type signature to include array arg
                    old_sig = calltypes.pop(rhs)
                    # argsort requires kws for typing so sig.args can't be used
                    # reusing sig.args since some types become Const in sig
                    argtyps = old_sig.args[: len(rhs.args)]
                    kwtyps = {name: typemap[v.name] for name, v in rhs.kws}
                    # Bodo Change: Try all possible implementations and then update
                    # the typemap
                    func_typs = impl_map[rhs.func.name]
                    call_type = None
                    for func_typ in func_typs:
                        try:
                            call_type = func_typ.get_call_type(
                                typingctx, [typemap[arr.name]] + list(argtyps), kwtyps
                            )
                            # Update typemap and calltypes
                            typemap.pop(rhs.func.name)
                            typemap[rhs.func.name] = func_typ
                            calltypes[rhs] = call_type
                            break
                        except Exception:
                            # If an implementation doesn't match, continue
                            pass
                    if call_type is None:
                        raise TypeError(f"No valid template found for {rhs.func.name}")
                    rhs.args = [arr] + rhs.args

            new_body.append(stmt)
        block.body = new_body


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.ir_utils.canonicalize_array_math)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "b2200e9100613631cc554f4b640bc1181ba7cea0ece83630122d15b86941be2e"
    ):  # pragma: no cover
        warnings.warn("canonicalize_array_math has changed")

numba.core.ir_utils.canonicalize_array_math = canonicalize_array_math
# Clobber import in parfor
numba.parfors.parfor.canonicalize_array_math = canonicalize_array_math
# Clobber import in inline_closurecall
numba.core.inline_closurecall.canonicalize_array_math = canonicalize_array_math


#################   End Array Math Changes   #################

#################   Start Bytes Changes   ##################

# Bodo Change, Update Numpy_rules_ufunc to ignore Bytes from ArrayCompatible.
# The correct change is probably to update Bytes to no longer extend ArrayCompatible,
# but this would require a lot of changes to update every place that imports types.Bytes,
# so we will wait to fully fix this inside Numba.


def _Numpy_Rules_ufunc_handle_inputs(cls, ufunc, args, kws):
    """
    Process argument types to a given *ufunc*.
    Returns a (base types, explicit outputs, ndims, layout) tuple where:
    - `base types` is a tuple of scalar types for each input
    - `explicit outputs` is a tuple of explicit output types (arrays)
    - `ndims` is the number of dimensions of the loop and also of
        any outputs, explicit or implicit
    - `layout` is the layout for any implicit output to be allocated
    """
    nin = ufunc.nin
    nout = ufunc.nout
    nargs = ufunc.nargs

    # preconditions
    assert nargs == nin + nout

    if len(args) < nin:
        msg = "ufunc '{0}': not enough arguments ({1} found, {2} required)"
        raise TypingError(msg=msg.format(ufunc.__name__, len(args), nin))

    if len(args) > nargs:
        msg = "ufunc '{0}': too many arguments ({1} found, {2} maximum)"
        raise TypingError(msg=msg.format(ufunc.__name__, len(args), nargs))

    # Hack: Bodo change to not match on Bytes
    args = [
        a.as_array
        if (isinstance(a, types.ArrayCompatible) and not isinstance(a, types.Bytes))
        else a
        for a in args
    ]
    # Hack: Bodo change to not match on Bytes
    arg_ndims = [
        a.ndim
        if (isinstance(a, types.ArrayCompatible) and not isinstance(a, types.Bytes))
        else 0
        for a in args
    ]
    ndims = max(arg_ndims)

    # explicit outputs must be arrays (no explicit scalar return values supported)
    explicit_outputs = args[nin:]

    # all the explicit outputs must match the number max number of dimensions
    if not all(d == ndims for d in arg_ndims[nin:]):
        msg = "ufunc '{0}' called with unsuitable explicit output arrays."
        raise TypingError(msg=msg.format(ufunc.__name__))

    # Hack: Bodo change to not match on Bytes
    if not all(
        (
            isinstance(output, types.ArrayCompatible)
            and not isinstance(output, types.Bytes)
        )
        for output in explicit_outputs
    ):
        msg = "ufunc '{0}' called with an explicit output that is not an array"
        raise TypingError(msg=msg.format(ufunc.__name__))

    if not all(output.mutable for output in explicit_outputs):
        msg = "ufunc '{0}' called with an explicit output that is read-only"
        raise TypingError(msg=msg.format(ufunc.__name__))

    # Hack: Bodo change to not match on Bytes
    # find the kernel to use, based only in the input types (as does NumPy)
    base_types = [
        x.dtype
        if isinstance(x, types.ArrayCompatible) and not isinstance(x, types.Bytes)
        else x
        for x in args
    ]

    # Figure out the output array layout, if needed.
    layout = None
    if ndims > 0 and (len(explicit_outputs) < ufunc.nout):
        layout = "C"
        # Hack: Bodo change to not match on Bytes
        layouts = [
            x.layout
            if isinstance(x, types.ArrayCompatible) and not isinstance(x, types.Bytes)
            else ""
            for x in args
        ]

        # Prefer C contig if any array is C contig.
        # Next, prefer F contig.
        # Defaults to C contig if not layouts are C/F.
        if "C" not in layouts and "F" in layouts:
            layout = "F"

    return base_types, explicit_outputs, ndims, layout


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(
        numba.core.typing.npydecl.Numpy_rules_ufunc._handle_inputs
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "4b97c64ad9c3d50e082538795054f35cf6d2fe962c3ca40e8377a4601b344d5c"
    ):  # pragma: no cover
        warnings.warn("Numpy_rules_ufunc._handle_inputs has changed")

numba.core.typing.npydecl.Numpy_rules_ufunc._handle_inputs = (
    _Numpy_Rules_ufunc_handle_inputs
)
numba.np.ufunc.dufunc.npydecl.Numpy_rules_ufunc._handle_inputs = (
    _Numpy_Rules_ufunc_handle_inputs
)

#################   End Bytes Changes   ##################


######### changes for removing unnecessary list limitations in Numba ######


def DictType__init__(self, keyty, valty, initial_value=None):
    from numba.types import (
        DictType,
        InitialValue,
        NoneType,
        Optional,
        Tuple,
        TypeRef,
        unliteral,
    )

    assert not isinstance(keyty, TypeRef)
    assert not isinstance(valty, TypeRef)
    keyty = unliteral(keyty)
    valty = unliteral(valty)
    if isinstance(keyty, (Optional, NoneType)):
        fmt = "Dict.key_type cannot be of type {}"
        raise TypingError(fmt.format(keyty))
    if isinstance(valty, (Optional, NoneType)):
        fmt = "Dict.value_type cannot be of type {}"
        raise TypingError(fmt.format(valty))
    # Bodo change: avoid unnecessary list/set restriction
    # _sentry_forbidden_types(keyty, valty)
    self.key_type = keyty
    self.value_type = valty
    self.keyvalue_type = Tuple([keyty, valty])
    name = "{}[{},{}]<iv={}>".format(
        self.__class__.__name__, keyty, valty, initial_value
    )
    super(DictType, self).__init__(name)
    InitialValue.__init__(self, initial_value)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.types.containers.DictType.__init__)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "475acd71224bd51526750343246e064ff071320c0d10c17b8b8ac81d5070d094"
    ):  # pragma: no cover
        warnings.warn("DictType.__init__ has changed")


numba.core.types.containers.DictType.__init__ = DictType__init__


def _legalize_arg_types(self, args):
    for i, a in enumerate(args, start=1):
        # Bodo change: avoid unnecessary list restriction
        # if isinstance(a, types.List):
        #     msg = (
        #         'Does not support list type inputs into '
        #         'with-context for arg {}'
        #     )
        #     raise errors.TypingError(msg.format(i))
        if isinstance(a, types.Dispatcher):
            msg = (
                "Does not support function type inputs into " "with-context for arg {}"
            )
            raise errors.TypingError(msg.format(i))


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(
        numba.core.dispatcher.ObjModeLiftedWith._legalize_arg_types
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "4793f44ebc7da8843e8f298e08cd8a5428b4b84b89fd9d5c650273fdb8fee5ee"
    ):  # pragma: no cover
        warnings.warn("ObjModeLiftedWith._legalize_arg_types has changed")


numba.core.dispatcher.ObjModeLiftedWith._legalize_arg_types = _legalize_arg_types


######### End changes for removing unnecessary list limitations in Numba ######


#########   Changes for Caching  #########

# TODO: [BE-1356] Remove these changes


def _overload_template_get_impl(self, args, kws):
    """Get implementation given the argument types.

    Returning a Dispatcher object.  The Dispatcher object is cached
    internally in `self._impl_cache`.
    """
    # Bodo change: Avoid looking at compiler flags. Inlining
    # and typing produce different compiler flags in certain places,
    # which leads to excessive compilation.
    cache_key = self.context, tuple(args), tuple(kws.items())
    try:
        impl, args = self._impl_cache[cache_key]
        return impl, args
    except KeyError:
        # pass and try outside the scope so as to not have KeyError with a
        # nested addition error in the case the _build_impl fails
        pass
    impl, args = self._build_impl(cache_key, args, kws)
    return impl, args


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(
        numba.core.typing.templates._OverloadFunctionTemplate._get_impl
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "4e27d07b214ca16d6e8ed88f70d886b6b095e160d8f77f8df369dd4ed2eb3fae"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.typing.templates._OverloadFunctionTemplate._get_impl has changed"
        )


numba.core.typing.templates._OverloadFunctionTemplate._get_impl = (
    _overload_template_get_impl
)


#########   End Changes for Caching  #########

#########   Start changes to improve parfor dead code elimination #########
def remove_dead_parfor(
    parfor, lives, lives_n_aliases, arg_aliases, alias_map, func_ir, typemap
):
    """remove dead code inside parfor including get/sets.

    Bodo Changes:
        - Remove setna if the result is unused because there are no cross-iteration dependencies
        - Refactor dead code elimination to convert branches of identical blocks to a jump, to enable
            removing more code such as branch conditions that are otherwise unused.
    """
    from numba.core.analysis import (
        compute_cfg_from_blocks,
        compute_live_map,
        compute_use_defs,
    )
    from numba.core.ir_utils import find_topo_order
    from numba.parfors.parfor import (
        _add_liveness_return_block,
        _update_parfor_get_setitems,
        dummy_return_in_loop_body,
        get_index_var,
        remove_dead_parfor_recursive,
        simplify_parfor_body_CFG,
    )

    with dummy_return_in_loop_body(parfor.loop_body):
        labels = find_topo_order(parfor.loop_body)

    # get/setitem replacement should ideally use dataflow to propagate setitem
    # saved values, but for simplicity we handle the common case of propagating
    # setitems in the first block (which is dominant) if the array is not
    # potentially changed in any way
    first_label = labels[0]
    first_block_saved_values = {}
    _update_parfor_get_setitems(
        parfor.loop_body[first_label].body,
        parfor.index_var,
        alias_map,
        first_block_saved_values,
        lives_n_aliases,
    )

    # remove saved first block setitems if array potentially changed later
    saved_arrs = set(first_block_saved_values.keys())
    for l in labels:
        if l == first_label:
            continue
        for stmt in parfor.loop_body[l].body:
            if (
                isinstance(stmt, ir.Assign)
                and isinstance(stmt.value, ir.Expr)
                and stmt.value.op == "getitem"
                and stmt.value.index.name == parfor.index_var.name
            ):
                continue
            varnames = set(v.name for v in stmt.list_vars())
            rm_arrs = varnames & saved_arrs
            for a in rm_arrs:
                first_block_saved_values.pop(a, None)

    # replace getitems with available value
    # e.g. A[i] = v; ... s = A[i]  ->  s = v
    for l in labels:
        if l == first_label:
            continue
        block = parfor.loop_body[l]
        saved_values = first_block_saved_values.copy()
        _update_parfor_get_setitems(
            block.body, parfor.index_var, alias_map, saved_values, lives_n_aliases
        )

    # after getitem replacement, remove extra setitems
    blocks = parfor.loop_body.copy()  # shallow copy is enough
    last_label = max(blocks.keys())
    return_label, tuple_var = _add_liveness_return_block(
        blocks, lives_n_aliases, typemap
    )
    # jump to return label
    jump = ir.Jump(return_label, ir.Loc("parfors_dummy", -1))
    blocks[last_label].body.append(jump)
    cfg = compute_cfg_from_blocks(blocks)
    usedefs = compute_use_defs(blocks)
    live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
    alias_set = set(alias_map.keys())

    for label, block in blocks.items():
        new_body = []
        in_lives = {v.name for v in block.terminator.list_vars()}
        # find live variables at the end of block
        for out_blk, _data in cfg.successors(label):
            in_lives |= live_map[out_blk]
        for stmt in reversed(block.body):
            # aliases of lives are also live for setitems
            alias_lives = in_lives & alias_set
            for v in alias_lives:
                in_lives |= alias_map[v]
            if (
                isinstance(stmt, (ir.StaticSetItem, ir.SetItem))
                and get_index_var(stmt).name == parfor.index_var.name
                and stmt.target.name not in in_lives
                and stmt.target.name not in arg_aliases
            ):
                continue
            # Bodo Change:
            # If a statement is a function call that normally can be removed only if
            # arg is no longer live, but no cross iteration side effects,
            # it can be safely removed if it is only live inside the loop.
            elif (
                isinstance(stmt, ir.Assign)
                and isinstance(stmt.value, ir.Expr)
                and stmt.value.op == "call"
            ):
                fdef = guard(find_callname, func_ir, stmt.value)
                # So far we only support setna
                if (
                    fdef == ("setna", "bodo.libs.array_kernels")
                    and stmt.value.args[0].name not in in_lives
                    and stmt.value.args[0].name not in arg_aliases
                ):
                    continue

            in_lives |= {v.name for v in stmt.list_vars()}
            new_body.append(stmt)
        new_body.reverse()
        block.body = new_body

    typemap.pop(tuple_var.name)  # remove dummy tuple type
    blocks[last_label].body.pop()  # remove jump

    # Bodo Change: Helper function to replace branches with jump if blocks are empty
    def trim_empty_parfor_branches(parfor):
        """
        Iterate through the parfor and replaces branches where the true
        and false labels lead to the same location and replaces them with a
        jump to one of the blocks. This is only implemented for only 2 cases:

        1. If both the truebr and falsebr are a single jump to the same
           location. For example:

            branch $60pred.106, 73, 77               ['$60pred.106']
        label 73:
            jump 125                                 []
        label 77:
            jump 125                                 []

        is replaced with

            jump 125                                 []
        label 73:
            jump 125                                 []
        label 77:
            jump 125                                 []

        2. If one of the blocks is a single jump that jumps to the
           other block. For example:

            branch $108pred.467, 384, 404            ['$108pred.467']
        label 384:
            jump 404                                 []
        label 404:
            return $48return_value.21                ['$48return_value.21']

        is replaced with

            jump 404                                 []
        label 384:
            jump 404                                 []
        label 404:
            return $48return_value.21                ['$48return_value.21']

        """
        changed = False
        blocks = parfor.loop_body.copy()
        for label, block in blocks.items():
            # Only look at the last statment in a block
            if len(block.body):
                end_stmt = block.body[-1]
                if isinstance(end_stmt, ir.Branch):
                    # If true/false conditions jump to blocks with a single jump to the same location,
                    # we can replace the branch with that instruction directly.
                    if (
                        len(blocks[end_stmt.truebr].body) == 1
                        and len(blocks[end_stmt.falsebr].body) == 1
                    ):
                        true_stmt = blocks[end_stmt.truebr].body[0]
                        false_stmt = blocks[end_stmt.falsebr].body[0]
                        if (
                            isinstance(true_stmt, ir.Jump)
                            and isinstance(false_stmt, ir.Jump)
                            and true_stmt.target == false_stmt.target
                        ):
                            parfor.loop_body[label].body[-1] = ir.Jump(
                                true_stmt.target, end_stmt.loc
                            )
                            changed = True
                    # If either block is just a jump to the other block we can remove the branch
                    elif len(blocks[end_stmt.truebr].body) == 1:
                        true_stmt = blocks[end_stmt.truebr].body[0]
                        if (
                            isinstance(true_stmt, ir.Jump)
                            and true_stmt.target == end_stmt.falsebr
                        ):
                            parfor.loop_body[label].body[-1] = ir.Jump(
                                true_stmt.target, end_stmt.loc
                            )
                            changed = True
                    elif len(blocks[end_stmt.falsebr].body) == 1:
                        false_stmt = blocks[end_stmt.falsebr].body[0]
                        if (
                            isinstance(false_stmt, ir.Jump)
                            and false_stmt.target == end_stmt.truebr
                        ):
                            parfor.loop_body[label].body[-1] = ir.Jump(
                                false_stmt.target, end_stmt.loc
                            )
                            changed = True

        return changed

    # End Bodo change

    # Bodo Change: Continue doing dead code elimination so long as the control flow
    # changes + remove unused blocks.
    changed = True
    while changed:
        """
        Process parfor body recursively.
        Note that this is the only place in this function that uses the
        argument lives instead of lives_n_aliases.  The former does not
        include the aliases of live variables but only the live variable
        names themselves.  See a comment in this function for how that
        is used.
        """
        remove_dead_parfor_recursive(
            parfor, lives, arg_aliases, alias_map, func_ir, typemap
        )

        # Simplify the CFG to squash any unused blocks .
        simplify_parfor_body_CFG(func_ir.blocks)

        # Prune blocks that are empty to allow eliminating parfor
        # branches that go to empty blocks. Its only possible to
        # do more dead code elimination if the control flow has
        # changed.
        changed = trim_empty_parfor_branches(parfor)

    # End Bodo change

    # remove parfor if empty
    is_empty = len(parfor.init_block.body) == 0
    for block in parfor.loop_body.values():
        is_empty &= len(block.body) == 0
    if is_empty:
        return None
    return parfor


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.parfors.parfor.remove_dead_parfor)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "1c9b008a7ead13e988e1efe67618d8f87f0b9f3d092cc2cd6bfcd806b1fdb859"
    ):
        warnings.warn("remove_dead_parfor has changed")


numba.parfors.parfor.remove_dead_parfor = remove_dead_parfor
numba.core.ir_utils.remove_dead_extensions[
    numba.parfors.parfor.Parfor
] = remove_dead_parfor


def simplify_parfor_body_CFG(blocks):
    """simplify CFG of body loops in parfors
    Bodo Change: Eliminate dead branches and an extra constant
    (since it won't exist in the type map).
    """
    from numba.core.analysis import compute_cfg_from_blocks
    from numba.core.ir_utils import simplify_CFG
    from numba.parfors.parfor import Parfor

    n_parfors = 0
    for block in blocks.values():
        for stmt in block.body:
            if isinstance(stmt, Parfor):
                n_parfors += 1
                parfor = stmt
                # add dummy return to enable CFG creation
                # can't use dummy_return_in_loop_body since body changes
                last_block = parfor.loop_body[max(parfor.loop_body.keys())]
                scope = last_block.scope
                loc = ir.Loc("parfors_dummy", -1)
                const = ir.Var(scope, mk_unique_var("$const"), loc)
                last_block.body.append(ir.Assign(ir.Const(0, loc), const, loc))
                last_block.body.append(ir.Return(const, loc))
                # Bodo change:
                # Eliminate any dead blocks as they will break the cfg. Normally
                # we won't encoutner dead blocks, but when we change the control
                # flow we might make a block unreachable.
                cfg = compute_cfg_from_blocks(parfor.loop_body)
                for dead in cfg.dead_nodes():
                    del parfor.loop_body[dead]
                # End Bodo change

                parfor.loop_body = simplify_CFG(parfor.loop_body)
                # The constant and return value  we added
                # above will be located in the last block.
                # We should remove both here.
                last_block = parfor.loop_body[max(parfor.loop_body.keys())]
                # Delete the return
                last_block.body.pop()
                # Bodo Change:
                # Delete the constant created above.
                last_block.body.pop()
                # End Bodo change
                # call on body recursively
                simplify_parfor_body_CFG(parfor.loop_body)
    return n_parfors


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.parfors.parfor.simplify_parfor_body_CFG)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "437ae96a5e8ec64a2b69a4f23ba8402a1d170262a5400aa0aa7bfe59e03bf726"
    ):
        warnings.warn("simplify_parfor_body_CFG has changed")


numba.parfors.parfor.simplify_parfor_body_CFG = simplify_parfor_body_CFG


#########   End changes to improve parfor dead code elimination   #########


#########  Start changes to Fix Numba objmode bug during caching  #########
# see https://github.com/numba/numba/issues/7572


def _lifted_compile(self, sig):
    import numba.core.event as ev
    from numba.core import compiler, sigutils
    from numba.core.compiler_lock import global_compiler_lock
    from numba.core.ir_utils import remove_dels

    with ExitStack() as scope:
        cres = None

        def cb_compiler(dur):
            if cres is not None:
                self._callback_add_compiler_timer(dur, cres)

        def cb_llvm(dur):
            if cres is not None:
                self._callback_add_llvm_timer(dur, cres)

        scope.enter_context(ev.install_timer("numba:compiler_lock", cb_compiler))
        scope.enter_context(ev.install_timer("numba:llvm_lock", cb_llvm))
        scope.enter_context(global_compiler_lock)

        # Use counter to track recursion compilation depth
        with self._compiling_counter:
            # XXX this is mostly duplicated from Dispatcher.
            flags = self.flags
            args, return_type = sigutils.normalize_signature(sig)

            # Don't recompile if signature already exists
            # (e.g. if another thread compiled it before we got the lock)
            existing = self.overloads.get(tuple(args))
            if existing is not None:
                return existing.entry_point

            self._pre_compile(args, return_type, flags)

            # Clone IR to avoid (some of the) mutation in the rewrite pass
            # Bodo change: avoid copy()
            cloned_func_ir = self.func_ir  # .copy()

            ev_details = dict(
                dispatcher=self,
                args=args,
                return_type=return_type,
            )
            with ev.trigger_event("numba:compile", data=ev_details):
                cres = compiler.compile_ir(
                    typingctx=self.typingctx,
                    targetctx=self.targetctx,
                    func_ir=cloned_func_ir,
                    args=args,
                    return_type=return_type,
                    flags=flags,
                    locals=self.locals,
                    lifted=(),
                    lifted_from=self.lifted_from,
                    is_lifted_loop=True,
                )

                # Check typing error if object mode is used
                if cres.typing_error is not None and not flags.enable_pyobject:
                    raise cres.typing_error
                self.add_overload(cres)

            # Bodo change: remove dels
            remove_dels(self.func_ir.blocks)
            return cres.entry_point


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.dispatcher.LiftedCode.compile)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "1351ebc5d8812dc8da167b30dad30eafb2ca9bf191b49aaed6241c21e03afff1"
    ):  # pragma: no cover
        warnings.warn("numba.core.dispatcher.LiftedCode.compile has changed")


numba.core.dispatcher.LiftedCode.compile = _lifted_compile


def compile_ir(
    typingctx,
    targetctx,
    func_ir,
    args,
    return_type,
    flags,
    locals,
    lifted=(),
    lifted_from=None,
    is_lifted_loop=False,
    library=None,
    pipeline_class=Compiler,
):
    """
    Compile a function with the given IR.

    For internal use only.
    """

    # This is a special branch that should only run on IR from a lifted loop
    if is_lifted_loop:
        # This code is pessimistic and costly, but it is a not often trodden
        # path and it will go away once IR is made immutable. The problem is
        # that the rewrite passes can mutate the IR into a state that makes
        # it possible for invalid tokens to be transmitted to lowering which
        # then trickle through into LLVM IR and causes RuntimeErrors as LLVM
        # cannot compile it. As a result the following approach is taken:
        # 1. Create some new flags that copy the original ones but switch
        #    off rewrites.
        # 2. Compile with 1. to get a compile result
        # 3. Try and compile another compile result but this time with the
        #    original flags (and IR being rewritten).
        # 4. If 3 was successful, use the result, else use 2.

        # create flags with no rewrites
        norw_flags = copy.deepcopy(flags)
        norw_flags.no_rewrites = True

        def compile_local(the_ir, the_flags):
            pipeline = pipeline_class(
                typingctx, targetctx, library, args, return_type, the_flags, locals
            )
            return pipeline.compile_ir(
                func_ir=the_ir, lifted=lifted, lifted_from=lifted_from
            )

        # compile with rewrites off, IR shouldn't be mutated irreparably
        # Bodo change: avoid func_ir.copy()
        norw_cres = compile_local(func_ir, norw_flags)

        # try and compile with rewrites on if no_rewrites was not set in the
        # original flags, IR might get broken but we've got a CompileResult
        # that's usable from above.
        rw_cres = None
        if not flags.no_rewrites:
            # Suppress warnings in compilation retry
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", errors.NumbaWarning)
                try:
                    # Bodo change: avoid func_ir.copy()
                    rw_cres = compile_local(func_ir, flags)
                except Exception:
                    pass
        # if the rewrite variant of compilation worked, use it, else use
        # the norewrites backup
        if rw_cres is not None:
            cres = rw_cres
        else:
            cres = norw_cres
        return cres

    else:
        pipeline = pipeline_class(
            typingctx, targetctx, library, args, return_type, flags, locals
        )
        return pipeline.compile_ir(
            func_ir=func_ir, lifted=lifted, lifted_from=lifted_from
        )


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.compiler.compile_ir)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "c48ce5493f4c43326e8cbdd46f3ea038b2b9045352d9d25894244798388e5e5b"
    ):  # pragma: no cover
        warnings.warn("numba.core.compiler.compile_ir has changed")


numba.core.compiler.compile_ir = compile_ir

#########  End changes to Fix Numba objmode bug during caching  #########


#########  Start changes to keep references to large const global arrays  #########
# To get const value for a large global array, Numba just uses the address of data
# pointer instead of embedding the whole array. However, the array needs to be alive
# during execution time (segfaults otherwise).
# We keep a reference to the array in the overload metadata to keep it
# alive. To reproduce the issue, use read_csv() with large number of columns (>50k).


def make_constant_array(self, builder, typ, ary):
    """
    Create an array structure reifying the given constant array.
    A low-level contiguous array constant is created in the LLVM IR.
    """
    import math

    from llvmlite import ir as lir

    datatype = self.get_data_type(typ.dtype)
    # Bodo change: change size limit to 10MB
    # don't freeze ary of non-contig or bigger than 10MB
    size_limit = 10**7

    if self.allow_dynamic_globals and (
        typ.layout not in "FC" or ary.nbytes > size_limit
    ):
        # get pointer from the ary
        dataptr = ary.ctypes.data
        data = self.add_dynamic_addr(builder, dataptr, info=str(type(dataptr)))
        rt_addr = self.add_dynamic_addr(builder, id(ary), info=str(type(ary)))
        # Bodo change: save constant array in target context to be added to function
        # overload metadata later
        self.global_arrays.append(ary)
    else:
        # Handle data: reify the flattened array in "C" or "F" order as a
        # global array of bytes.
        flat = ary.flatten(order=typ.layout)
        # Bodo change: dt64/td64 fail in bytearray so cast to int64
        if isinstance(typ.dtype, (types.NPDatetime, types.NPTimedelta)):
            flat = flat.view("int64")
        # Note: we use `bytearray(flat.data)` instead of `bytearray(flat)` to
        #       workaround issue #1850 which is due to numpy issue #3147
        # TODO: Replace with cgutils.create_constant_array when Numba 0.56 merges.
        val = bytearray(flat.data)
        consts = lir.Constant(lir.ArrayType(lir.IntType(8), len(val)), val)
        data = cgutils.global_constant(builder, ".const.array.data", consts)
        # Ensure correct data alignment (issue #1933)
        data.align = self.get_abi_alignment(datatype)
        # No reference to parent ndarray
        rt_addr = None

    # Handle shape
    llintp = self.get_value_type(types.intp)
    shapevals = [self.get_constant(types.intp, s) for s in ary.shape]
    # TODO: Replace with cgutils.create_constant_array when Numba 0.56 merges.
    cshape = lir.Constant(lir.ArrayType(llintp, len(shapevals)), shapevals)

    # Handle strides
    stridevals = [self.get_constant(types.intp, s) for s in ary.strides]
    # TODO: Replace with cgutils.create_constant_array when Numba 0.56 merges.
    cstrides = lir.Constant(lir.ArrayType(llintp, len(stridevals)), stridevals)

    intp_itemsize = self.get_constant(types.intp, ary.dtype.itemsize)

    # Bodo change: create a LiteralStruct that can be embedded in constant globals
    # instead of calling populate_array (which creates a regular struct reference)

    nitems = self.get_constant(types.intp, math.prod(ary.shape))

    # create a literal struct that matches data model of arrays in Numba:
    # https://github.com/numba/numba/blob/0499b906a850af34f0e2fdcc6b3b3836cdc95297/numba/core/datamodel/models.py#L862
    return lir.Constant.literal_struct(
        [
            # meminfo
            self.get_constant_null(types.MemInfoPointer(typ.dtype)),
            # parent
            self.get_constant_null(types.pyobject),
            # nitems
            nitems,
            # itemsize
            intp_itemsize,
            # data
            data.bitcast(self.get_value_type(types.CPointer(typ.dtype))),
            # shape
            cshape,
            # strides
            cstrides,
        ]
    )


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.base.BaseContext.make_constant_array)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "5721b5360b51f782f79bd794f7bf4d48657911ecdc05c30db22fd55f15dad821"
    ):  # pragma: no cover
        warnings.warn("numba.core.base.BaseContext.make_constant_array has changed")


numba.core.base.BaseContext.make_constant_array = make_constant_array


# Bodo change: avoid incref/decref if the meminfo is a constant global (writing to
# constant globals can lead to segfault)
def _define_atomic_inc_dec(module, op, ordering):
    """Define a llvm function for atomic increment/decrement to the given module
    Argument ``op`` is the operation "add"/"sub".  Argument ``ordering`` is
    the memory ordering.  The generated function returns the new value.
    """
    from llvmlite import ir as lir
    from numba.core.runtime.nrtdynmod import _word_type

    ftype = lir.FunctionType(_word_type, [_word_type.as_pointer()])
    fn_atomic = lir.Function(module, ftype, name="nrt_atomic_{0}".format(op))

    [ptr] = fn_atomic.args
    bb = fn_atomic.append_basic_block()
    builder = lir.IRBuilder(bb)
    ONE = lir.Constant(_word_type, 1)
    # Bodo change: disable atomic incref/decref since we don't use threading
    if False:  # pragma: no cover
        oldval = builder.atomic_rmw(op, ptr, ONE, ordering=ordering)
        # Perform the operation on the old value so that we can pretend returning
        # the "new" value.
        res = getattr(builder, op)(oldval, ONE)
        builder.ret(res)
    else:
        oldval = builder.load(ptr)
        newval = getattr(builder, op)(oldval, ONE)
        # Bodo change: store value only if not a constant global. we set refcount = -1
        # in case of globals (see lower_constant of StringArrayType)
        is_not_const = builder.icmp_signed("!=", oldval, lir.Constant(oldval.type, -1))
        with cgutils.if_likely(builder, is_not_const):
            builder.store(newval, ptr)
        # Bodo change: fix a bug in Numba code that returns old value instead of new
        # (as expected by _define_nrt_decref)
        builder.ret(newval)

    return fn_atomic


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.runtime.nrtdynmod._define_atomic_inc_dec)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "9cc02c532b2980b6537b702f5608ea603a1ff93c6d3c785ae2cf48bace273f48"
    ):
        warnings.warn("numba.core.runtime.nrtdynmod._define_atomic_inc_dec has changed")


numba.core.runtime.nrtdynmod._define_atomic_inc_dec = _define_atomic_inc_dec


def NativeLowering_run_pass(self, state):
    from llvmlite import binding as llvm
    from numba.core import funcdesc, lowering
    from numba.core.typed_passes import fallback_context

    if state.library is None:
        codegen = state.targetctx.codegen()
        state.library = codegen.create_library(state.func_id.func_qualname)
        # Enable object caching upfront, so that the library can
        # be later serialized.
        state.library.enable_object_caching()

    library = state.library
    targetctx = state.targetctx
    interp = state.func_ir  # why is it called this?!
    typemap = state.typemap
    restype = state.return_type
    calltypes = state.calltypes
    flags = state.flags
    metadata = state.metadata
    pre_stats = llvm.passmanagers.dump_refprune_stats()

    msg = "Function %s failed at nopython " "mode lowering" % (state.func_id.func_name,)
    with fallback_context(state, msg):
        # Lowering
        fndesc = funcdesc.PythonFunctionDescriptor.from_specialized_function(
            interp,
            typemap,
            restype,
            calltypes,
            mangler=targetctx.mangler,
            inline=flags.forceinline,
            noalias=flags.noalias,
            abi_tags=[flags.get_mangle_string()],
        )

        # Bodo change: save constant global arrays to be used below
        targetctx.global_arrays = []
        with targetctx.push_code_library(library):
            lower = lowering.Lower(
                targetctx, library, fndesc, interp, metadata=metadata
            )
            lower.lower()
            if not flags.no_cpython_wrapper:
                lower.create_cpython_wrapper(flags.release_gil)

            if not flags.no_cfunc_wrapper:
                # skip cfunc wrapper generation if unsupported
                # argument or return types are used
                for t in state.args:
                    if isinstance(t, (types.Omitted, types.Generator)):
                        break
                else:
                    if isinstance(restype, (types.Optional, types.Generator)):
                        pass
                    else:
                        lower.create_cfunc_wrapper()

            env = lower.env
            call_helper = lower.call_helper
            del lower

        from numba.core.compiler import _LowerResult  # TODO: move this

        if flags.no_compile:
            state["cr"] = _LowerResult(fndesc, call_helper, cfunc=None, env=env)
        else:
            # Prepare for execution
            # Insert native function for use by other jitted-functions.
            # We also register its library to allow for inlining.
            cfunc = targetctx.get_executable(library, fndesc, env)
            targetctx.insert_user_function(cfunc, fndesc, [library])
            state["cr"] = _LowerResult(fndesc, call_helper, cfunc=cfunc, env=env)

        # Bodo change: save constant global arrays in overload metadata so they are not
        # garbage collected before execution
        metadata["global_arrs"] = targetctx.global_arrays
        targetctx.global_arrays = []
        # capture pruning stats
        post_stats = llvm.passmanagers.dump_refprune_stats()
        metadata["prune_stats"] = post_stats - pre_stats

        # Save the LLVM pass timings
        metadata["llvm_pass_timings"] = library.recorded_timings
    return True


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.typed_passes.NativeLowering.run_pass)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "a777ce6ce1bb2b1cbaa3ac6c2c0e2adab69a9c23888dff5f1cbb67bfb176b5de"
    ):  # pragma: no cover
        warnings.warn("numba.core.typed_passes.NativeLowering.run_pass has changed")


numba.core.typed_passes.NativeLowering.run_pass = NativeLowering_run_pass


#########  End changes to keep references to large const global arrays  #########


#########  Start changes to allow lists of optional values unboxing  #########
# change types.List unboxing to support optional values, see test_match_groups


def _python_list_to_native(typ, obj, c, size, listptr, errorptr):
    """
    Construct a new native list from a Python list.
    """
    from llvmlite import ir as lir
    from numba.core.boxing import _NumbaTypeHelper
    from numba.cpython import listobj

    def check_element_type(nth, itemobj, expected_typobj):
        typobj = nth.typeof(itemobj)
        # Check if *typobj* is NULL
        with c.builder.if_then(
            cgutils.is_null(c.builder, typobj),
            likely=False,
        ):
            c.builder.store(cgutils.true_bit, errorptr)
            loop.do_break()
        # Mandate that objects all have the same exact type
        type_mismatch = c.builder.icmp_signed("!=", typobj, expected_typobj)

        # Bodo change: avoid typecheck for Optional types since it fails
        # TODO(ehsan): add infrastructure for proper type check for Optional
        if not isinstance(typ.dtype, types.Optional):
            with c.builder.if_then(type_mismatch, likely=False):
                c.builder.store(cgutils.true_bit, errorptr)
                c.pyapi.err_format(
                    "PyExc_TypeError",
                    "can't unbox heterogeneous list: %S != %S",
                    expected_typobj,
                    typobj,
                )
                c.pyapi.decref(typobj)
                loop.do_break()
        c.pyapi.decref(typobj)

    # Allocate a new native list
    ok, list = listobj.ListInstance.allocate_ex(c.context, c.builder, typ, size)
    with c.builder.if_else(ok, likely=True) as (if_ok, if_not_ok):
        with if_ok:
            list.size = size
            zero = lir.Constant(size.type, 0)
            with c.builder.if_then(c.builder.icmp_signed(">", size, zero), likely=True):
                # Traverse Python list and unbox objects into native list
                with _NumbaTypeHelper(c) as nth:
                    # Note: *expected_typobj* can't be NULL
                    expected_typobj = nth.typeof(c.pyapi.list_getitem(obj, zero))
                    with cgutils.for_range(c.builder, size) as loop:
                        itemobj = c.pyapi.list_getitem(obj, loop.index)
                        check_element_type(nth, itemobj, expected_typobj)
                        # XXX we don't call native cleanup for each
                        # list element, since that would require keeping
                        # of which unboxings have been successful.
                        native = c.unbox(typ.dtype, itemobj)
                        with c.builder.if_then(native.is_error, likely=False):
                            c.builder.store(cgutils.true_bit, errorptr)
                            loop.do_break()
                        # The reference is borrowed so incref=False
                        list.setitem(loop.index, native.value, incref=False)
                    c.pyapi.decref(expected_typobj)
            if typ.reflected:
                list.parent = obj
            # Stuff meminfo pointer into the Python object for
            # later reuse.
            with c.builder.if_then(
                c.builder.not_(c.builder.load(errorptr)), likely=False
            ):
                c.pyapi.object_set_private_data(obj, list.meminfo)
            list.set_dirty(False)
            c.builder.store(list.value, listptr)

        with if_not_ok:
            c.builder.store(cgutils.true_bit, errorptr)

    # If an error occurred, drop the whole native list
    with c.builder.if_then(c.builder.load(errorptr)):
        c.context.nrt.decref(c.builder, typ, list.value)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.boxing._python_list_to_native)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "f8e546df8b07adfe74a16b6aafb1d4fddbae7d3516d7944b3247cc7c9b7ea88a"
    ):  # pragma: no cover
        warnings.warn("numba.core.boxing._python_list_to_native has changed")


numba.core.boxing._python_list_to_native = _python_list_to_native


#########  End changes to allow lists of optional values unboxing  #########


# change string constant lowering to use literal struct (to be able to use in other
# constant globals like Series)
def make_string_from_constant(context, builder, typ, literal_string):
    """
    Get string data by `compile_time_get_string_data()` and return a
    unicode_type LLVM value
    """
    from llvmlite import ir as lir
    from numba.cpython.hashing import _Py_hash_t
    from numba.cpython.unicode import compile_time_get_string_data

    databytes, length, kind, is_ascii, hashv = compile_time_get_string_data(
        literal_string
    )
    mod = builder.module
    gv = context.insert_const_bytes(mod, databytes)

    # Bodo change: use literal struct instead of struct proxy
    # literal struct that matches unicode data model:
    return lir.Constant.literal_struct(
        [
            # ('data', types.voidptr)
            gv,
            # ('length', types.intp)
            context.get_constant(types.intp, length),
            # ('kind', types.int32)
            context.get_constant(types.int32, kind),
            # ('is_ascii', types.uint32)
            context.get_constant(types.uint32, is_ascii),
            # ('hash', _Py_hash_t),
            # Set hash to -1 to indicate that it should be computed.
            # We cannot bake in the hash value because of hashseed randomization.
            context.get_constant(_Py_hash_t, -1),
            # ('meminfo', types.MemInfoPointer(types.voidptr))
            context.get_constant_null(types.MemInfoPointer(types.voidptr)),
            # ('parent', types.pyobject),
            context.get_constant_null(types.pyobject),
        ]
    )


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.cpython.unicode.make_string_from_constant)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "525bd507383e06152763e2f046dae246cd60aba027184f50ef0fc9a80d4cd7fa"
    ):
        warnings.warn("numba.cpython.unicode.make_string_from_constant has changed")


numba.cpython.unicode.make_string_from_constant = make_string_from_constant


#########  Start changes to allow IntEnum support for `np.empty`, `np.zeros`, and `np.ones`  #########
# change `parse_shape` to support IntEnum


def parse_shape(shape):
    """
    Given a shape, return the number of dimensions.
    """
    ndim = None
    if isinstance(shape, types.Integer):
        ndim = 1
    elif isinstance(shape, (types.Tuple, types.UniTuple)):
        # Bodo Change: support IntEnum members within tuples
        # see: https://github.com/Bodo-inc/Bodo/pull/3499
        if all(isinstance(s, (types.Integer, types.IntEnumMember)) for s in shape):
            ndim = len(shape)
    return ndim


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.typing.npydecl.parse_shape)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "e62e3ff09d36df5ac9374055947d6a8be27160ce32960d3ef6cb67f89bd16429"
    ):
        warnings.warn("numba.core.typing.npydecl.parse_shape has changed")

numba.core.typing.npydecl.parse_shape = parse_shape
#########  End changes to allow IntEnum support for `np.empty`, `np.zeros`, and `np.ones`  #########

#########  Start changes to allow unsupported types in `ShapeEquivSet._getnames()`  #########


def _get_names(self, obj):
    """Return a set of names for the given obj, where array and tuples
    are broken down to their individual shapes or elements. This is
    safe because both Numba array shapes and Python tuples are immutable.
    """
    if isinstance(obj, ir.Var) or isinstance(obj, str):
        name = obj if isinstance(obj, str) else obj.name
        if name not in self.typemap:
            return (name,)
        typ = self.typemap[name]
        if isinstance(typ, (types.BaseTuple, types.ArrayCompatible)):
            ndim = typ.ndim if isinstance(typ, types.ArrayCompatible) else len(typ)
            # Treat 0d array as if it were a scalar.
            if ndim == 0:
                return (name,)
            else:
                return tuple("{}#{}".format(name, i) for i in range(ndim))
        else:
            return (name,)
    elif isinstance(obj, ir.Const):
        if isinstance(obj.value, tuple):
            return obj.value
        else:
            return (obj.value,)
    elif isinstance(obj, tuple):

        def get_names(x):
            names = self._get_names(x)
            if len(names) != 0:
                return names[0]
            return names

        return tuple(get_names(x) for x in obj)
    elif isinstance(obj, int):
        return (obj,)
    # Bodo Change: removed error throwing
    return ()


def get_equiv_const(self, obj):
    """If the given object is equivalent to a constant scalar,
    return the scalar value, or None otherwise.
    """
    names = self._get_names(obj)
    # Bodo Change: accounted for the case where len(names) == 0
    if len(names) != 1:
        return None
    return super(numba.parfors.array_analysis.ShapeEquivSet, self).get_equiv_const(
        names[0]
    )


def get_equiv_set(self, obj):
    """Return the set of equivalent objects."""
    names = self._get_names(obj)
    # Bodo Change: accounted for the case where len(names) == 0
    if len(names) != 1:
        return None
    return super(numba.parfors.array_analysis.ShapeEquivSet, self).get_equiv_set(
        names[0]
    )


if _check_numba_change:  # pragma: no cover
    for name, orig, new, hash in (
        (
            "numba.parfors.array_analysis.ShapeEquivSet._get_names",
            numba.parfors.array_analysis.ShapeEquivSet._get_names,
            _get_names,
            "8c9bf136109028d5445fd0a82387b6abeb70c23b20b41e2b50c34ba5359516ee",
        ),
        (
            "numba.parfors.array_analysis.ShapeEquivSet.get_equiv_const",
            numba.parfors.array_analysis.ShapeEquivSet.get_equiv_const,
            get_equiv_const,
            "bef410ca31a9e29df9ee74a4a27d339cc332564e4a237828b8a4decf625ce44e",
        ),
        (
            "numba.parfors.array_analysis.ShapeEquivSet.get_equiv_set",
            numba.parfors.array_analysis.ShapeEquivSet.get_equiv_set,
            get_equiv_set,
            "ec936d340c488461122eb74f28a28b88227cb1f1bca2b9ba3c19258cfe1eb40a",
        ),
    ):
        lines = inspect.getsource(orig)
        if hashlib.sha256(lines.encode()).hexdigest() != hash:
            warnings.warn(f"{name} has changed")


numba.parfors.array_analysis.ShapeEquivSet._get_names = _get_names
numba.parfors.array_analysis.ShapeEquivSet.get_equiv_const = get_equiv_const
numba.parfors.array_analysis.ShapeEquivSet.get_equiv_set = get_equiv_set
#########  End changes to allow unsupported types in `ShapeEquivSet._getnames()`  #########


# Bodo change: avoid raising errors for constant global dictionaries
def raise_on_unsupported_feature(func_ir, typemap):  # pragma: no cover
    """
    Helper function to walk IR and raise if it finds op codes
    that are unsupported. Could be extended to cover IR sequences
    as well as op codes. Intended use is to call it as a pipeline
    stage just prior to lowering to prevent LoweringErrors for known
    unsupported features.
    """
    import numpy

    gdb_calls = []  # accumulate calls to gdb/gdb_init

    # issue 2195: check for excessively large tuples
    for arg_name in func_ir.arg_names:
        if (
            arg_name in typemap
            and isinstance(typemap[arg_name], types.containers.UniTuple)
            and typemap[arg_name].count > 1000
        ):
            # Raise an exception when len(tuple) > 1000. The choice of this number (1000)
            # was entirely arbitrary
            msg = (
                "Tuple '{}' length must be smaller than 1000.\n"
                "Large tuples lead to the generation of a prohibitively large "
                "LLVM IR which causes excessive memory pressure "
                "and large compile times.\n"
                "As an alternative, the use of a 'list' is recommended in "
                "place of a 'tuple' as lists do not suffer from this problem.".format(
                    arg_name
                )
            )
            raise errors.UnsupportedError(msg, func_ir.loc)

    for blk in func_ir.blocks.values():
        for stmt in blk.find_insts(ir.Assign):
            # This raises on finding `make_function`
            if isinstance(stmt.value, ir.Expr):
                if stmt.value.op == "make_function":
                    val = stmt.value

                    # See if the construct name can be refined
                    code = getattr(val, "code", None)
                    if code is not None:
                        # check if this is a closure, the co_name will
                        # be the captured function name which is not
                        # useful so be explicit
                        if getattr(val, "closure", None) is not None:
                            use = "<creating a function from a closure>"
                            expr = ""
                        else:
                            use = code.co_name
                            expr = "(%s) " % use
                    else:
                        use = "<could not ascertain use case>"
                        expr = ""

                    msg = (
                        "Numba encountered the use of a language "
                        "feature it does not support in this context: "
                        "%s (op code: make_function not supported). If "
                        "the feature is explicitly supported it is "
                        "likely that the result of the expression %s"
                        "is being used in an unsupported manner."
                    ) % (use, expr)
                    raise errors.UnsupportedError(msg, stmt.value.loc)

            # this checks for gdb initialization calls, only one is permitted
            if isinstance(stmt.value, (ir.Global, ir.FreeVar)):
                val = stmt.value
                val = getattr(val, "value", None)
                if val is None:
                    continue

                # check global function
                found = False
                if isinstance(val, pytypes.FunctionType):
                    found = val in {numba.gdb, numba.gdb_init}
                if not found:  # freevar bind to intrinsic
                    found = getattr(val, "_name", "") == "gdb_internal"
                if found:
                    gdb_calls.append(stmt.loc)  # report last seen location

            # this checks that np.<type> was called if view is called
            if isinstance(stmt.value, ir.Expr):
                if stmt.value.op == "getattr" and stmt.value.attr == "view":
                    var = stmt.value.value.name
                    if isinstance(typemap[var], types.Array):
                        continue
                    df = func_ir.get_definition(var)
                    cn = guard(find_callname, func_ir, df)
                    if cn and cn[1] == "numpy":
                        ty = getattr(numpy, cn[0])
                        if numpy.issubdtype(ty, numpy.integer) or numpy.issubdtype(
                            ty, numpy.floating
                        ):
                            continue

                    vardescr = "" if var.startswith("$") else "'{}' ".format(var)
                    raise TypingError(
                        "'view' can only be called on NumPy dtypes, "
                        "try wrapping the variable {}with 'np.<dtype>()'".format(
                            vardescr
                        ),
                        loc=stmt.loc,
                    )

            # checks for globals that are also reflected
            if isinstance(stmt.value, ir.Global):
                ty = typemap[stmt.target.name]
                msg = (
                    "The use of a %s type, assigned to variable '%s' in "
                    "globals, is not supported as globals are considered "
                    "compile-time constants and there is no known way to "
                    "compile a %s type as a constant."
                )
                # Bodo change: remove types.DictType, types.List
                if isinstance(ty, types.ListType):
                    raise TypingError(msg % (ty, stmt.value.name, ty), loc=stmt.loc)

            # checks for generator expressions (yield in use when func_ir has
            # not been identified as a generator).
            if isinstance(stmt.value, ir.Yield) and not func_ir.is_generator:
                msg = "The use of generator expressions is unsupported."
                raise errors.UnsupportedError(msg, loc=stmt.loc)

    # There is more than one call to function gdb/gdb_init
    if len(gdb_calls) > 1:
        msg = (
            "Calling either numba.gdb() or numba.gdb_init() more than once "
            "in a function is unsupported (strange things happen!), use "
            "numba.gdb_breakpoint() to create additional breakpoints "
            "instead.\n\nRelevant documentation is available here:\n"
            "https://numba.pydata.org/numba-doc/latest/user/troubleshoot.html"
            "/troubleshoot.html#using-numba-s-direct-gdb-bindings-in-"
            "nopython-mode\n\nConflicting calls found at:\n %s"
        )
        buf = "\n".join([x.strformat() for x in gdb_calls])
        raise errors.UnsupportedError(msg % buf)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.ir_utils.raise_on_unsupported_feature)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "237a4fe8395a40899279c718bc3754102cd2577463ef2f48daceea78d79b2d5e"
    ):  # pragma: no cover
        warnings.warn("numba.core.ir_utils.raise_on_unsupported_feature has changed")

numba.core.ir_utils.raise_on_unsupported_feature = raise_on_unsupported_feature
numba.core.typed_passes.raise_on_unsupported_feature = raise_on_unsupported_feature


# Support unboxing regular dictionaries as Numba's typed dictionary
# TODO(ehsan): move to Numba
# TODO(ehsan): reflection is not supported. Throw warning if dict is modified?
@typeof_impl.register(dict)
def _typeof_dict(val, c):
    if len(val) == 0:  # pragma: no cover
        raise ValueError("Cannot type empty dict")
    k, v = next(iter(val.items()))
    key_type = typeof_impl(k, c)
    value_type = typeof_impl(v, c)
    if key_type is None or value_type is None:  # pragma: no cover
        raise ValueError(f"Cannot type dict element type {type(k)}, {type(v)}")
    return types.DictType(key_type, value_type)


# replace Dict unboxing to support regular dictionaries as well
def unbox_dicttype(typ, val, c):
    from llvmlite import ir as lir
    from numba.typed import dictobject
    from numba.typed.typeddict import Dict

    context = c.context

    # Bodo change: check for regular dictionary by checking '_opaque' attribute which is
    # typed.Dict specific. If regular dict, convert to typed.Dict before unboxing
    valptr = cgutils.alloca_once_value(c.builder, val)
    has_opaque = c.pyapi.object_hasattr_string(val, "_opaque")
    is_regular_dict = c.builder.icmp_unsigned(
        "==", has_opaque, lir.Constant(has_opaque.type, 0)
    )

    kt = typ.key_type
    vt = typ.value_type

    def make_dict():
        return numba.typed.Dict.empty(kt, vt)

    def copy_dict(out_dict, in_dict):
        for k, v in in_dict.items():
            out_dict[k] = v

    with c.builder.if_then(is_regular_dict):
        # allocate a new typed.Dict and copy values
        make_dict_obj = c.pyapi.unserialize(c.pyapi.serialize_object(make_dict))
        dct_val = c.pyapi.call_function_objargs(make_dict_obj, [])
        copy_dict_obj = c.pyapi.unserialize(c.pyapi.serialize_object(copy_dict))
        c.pyapi.call_function_objargs(copy_dict_obj, [dct_val, val])
        c.builder.store(dct_val, valptr)

    val = c.builder.load(valptr)
    # done Bodo change

    # Check that `type(val) is Dict`
    dict_type = c.pyapi.unserialize(c.pyapi.serialize_object(Dict))
    valtype = c.pyapi.object_type(val)
    same_type = c.builder.icmp_unsigned("==", valtype, dict_type)

    with c.builder.if_else(same_type) as (then, orelse):
        with then:
            miptr = c.pyapi.object_getattr_string(val, "_opaque")

            mip_type = types.MemInfoPointer(types.voidptr)
            native = c.unbox(mip_type, miptr)

            mi = native.value

            argtypes = mip_type, typeof(typ)

            def convert(mi, typ):  # pragma: no cover
                return dictobject._from_meminfo(mi, typ)

            sig = signature(typ, *argtypes)
            nil_typeref = context.get_constant_null(argtypes[1])
            args = (mi, nil_typeref)
            is_error, dctobj = c.pyapi.call_jit_code(convert, sig, args)
            # decref here because we are stealing a reference.
            c.context.nrt.decref(c.builder, typ, dctobj)

            c.pyapi.decref(miptr)
            bb_unboxed = c.builder.basic_block

        with orelse:
            # Raise error on incorrect type
            c.pyapi.err_format(
                "PyExc_TypeError",
                "can't unbox a %S as a %S",
                valtype,
                dict_type,
            )
            bb_else = c.builder.basic_block

    # Phi nodes to gather the output
    dctobj_res = c.builder.phi(dctobj.type)
    is_error_res = c.builder.phi(is_error.type)

    dctobj_res.add_incoming(dctobj, bb_unboxed)
    dctobj_res.add_incoming(dctobj.type(None), bb_else)

    is_error_res.add_incoming(is_error, bb_unboxed)
    is_error_res.add_incoming(cgutils.true_bit, bb_else)

    # cleanup
    c.pyapi.decref(dict_type)
    c.pyapi.decref(valtype)

    # Bodo change: remove the typed.Dict object that is not necessary anymore
    with c.builder.if_then(is_regular_dict):
        c.pyapi.decref(val)

    return NativeValue(dctobj_res, is_error=is_error_res)


# Import numba.core.types.typeddict to replace DictType unboxing
# NOTE: this import triggers some compilation so it has to be at the end of this file
# to make sure all patches are applied first
import numba.typed.typeddict  # noqa

if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(
        numba.core.pythonapi._unboxers.functions[numba.core.types.DictType]
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "5f6f183b94dc57838538c668a54c2476576c85d8553843f3219f5162c61e7816"
    ):
        warnings.warn("unbox_dicttype has changed")
numba.core.pythonapi._unboxers.functions[types.DictType] = unbox_dicttype


#########  Start bytecode changes for Python 3.10  #########

# Remove once https://github.com/numba/numba/pull/7866
# and https://github.com/numba/numba/pull/7964/ merges

# Bodo Change: Add op_DICT_UPDATE to byteflow and interpreter


def op_DICT_UPDATE_byteflow(self, state, inst):
    value = state.pop()
    index = inst.arg
    target = state.peek(index)
    updatevar = state.make_temp()
    res = state.make_temp()
    state.append(inst, target=target, value=value, updatevar=updatevar, res=res)


if _check_numba_change:  # pragma: no cover
    if hasattr(numba.core.byteflow.TraceRunner, "op_DICT_UPDATE"):
        warnings.warn("numba.core.byteflow.TraceRunner.op_DICT_UPDATE has changed")


numba.core.byteflow.TraceRunner.op_DICT_UPDATE = op_DICT_UPDATE_byteflow


def op_DICT_UPDATE_interpreter(self, inst, target, value, updatevar, res):
    from numba.core import ir

    target = self.get(target)
    value = self.get(value)
    updateattr = ir.Expr.getattr(target, "update", loc=self.loc)
    self.store(value=updateattr, name=updatevar)
    updateinst = ir.Expr.call(self.get(updatevar), (value,), (), loc=self.loc)
    self.store(value=updateinst, name=res)


if _check_numba_change:  # pragma: no cover
    if hasattr(numba.core.interpreter.Interpreter, "op_DICT_UPDATE"):
        warnings.warn("numba.core.interpreter.Interpreter.op_DICT_UPDATE has changed")


numba.core.interpreter.Interpreter.op_DICT_UPDATE = op_DICT_UPDATE_interpreter

# Bodo Change: Support Dict.update as a fallback.
@numba.extending.overload_method(numba.core.types.DictType, "update")
def ol_dict_update(d, other):
    if not isinstance(d, numba.core.types.DictType):
        return
    if not isinstance(other, numba.core.types.DictType):
        return

    def impl(d, other):
        for k, v in other.items():
            d[k] = v

    return impl


if _check_numba_change:  # pragma: no cover
    if hasattr(numba.core.interpreter.Interpreter, "ol_dict_update"):
        warnings.warn("numba.typed.dictobject.ol_dict_update has changed")


def op_CALL_FUNCTION_EX_byteflow(self, state, inst):
    from numba.core.utils import PYVERSION

    # Bodo change enable varkwarg with PYVERSION 3.10
    if inst.arg & 1 and PYVERSION != (3, 10):
        errmsg = "CALL_FUNCTION_EX with **kwargs not supported"
        raise errors.UnsupportedError(errmsg)
    if inst.arg & 1:
        varkwarg = state.pop()
    else:
        varkwarg = None
    vararg = state.pop()
    func = state.pop()
    res = state.make_temp()
    state.append(inst, func=func, vararg=vararg, varkwarg=varkwarg, res=res)
    state.push(res)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.byteflow.TraceRunner.op_CALL_FUNCTION_EX)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "349e7cfd27f5dab80fe15a7728c5f098f3f225ba8512d84331e39d01e863c6d4"
    ):
        warnings.warn("numba.core.byteflow.TraceRunner.op_CALL_FUNCTION_EX has changed")
numba.core.byteflow.TraceRunner.op_CALL_FUNCTION_EX = op_CALL_FUNCTION_EX_byteflow


def op_CALL_FUNCTION_EX_interpreter(self, inst, func, vararg, varkwarg, res):
    func = self.get(func)
    vararg = self.get(vararg)
    # BODO CHANGE: Add varkwarg to the call
    if varkwarg is not None:
        varkwarg = self.get(varkwarg)
    expr = ir.Expr.call(func, [], [], loc=self.loc, vararg=vararg, varkwarg=varkwarg)
    self.store(expr, res)


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.interpreter.Interpreter.op_CALL_FUNCTION_EX)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "84846e5318ab7ccc8f9abaae6ab9e0ca879362648196f9d4b0ffb91cf2e01f5d"
    ):
        warnings.warn(
            "numba.core.interpreter.Interpreter.op_CALL_FUNCTION_EX has changed"
        )
numba.core.interpreter.Interpreter.op_CALL_FUNCTION_EX = op_CALL_FUNCTION_EX_interpreter


@classmethod
def ir_expr_call(cls, func, args, kws, loc, vararg=None, varkwarg=None, target=None):
    assert isinstance(func, ir.Var)
    assert isinstance(loc, ir.Loc)
    op = "call"
    # BODO CHANGE: Include the varkwarg argument
    return cls(
        op=op,
        loc=loc,
        func=func,
        args=args,
        kws=kws,
        vararg=vararg,
        varkwarg=varkwarg,
        target=target,
    )


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(ir.Expr.call)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "665601d0548d4f648d454492e542cb8aa241107a8df6bc68d0eec664c9ada738"
    ):
        warnings.warn("ir.Expr.call has changed")

ir.Expr.call = ir_expr_call


# Bodo Change: Include bytecode changes in the Numba pipeline. Necessary for
# internally compiled functions.
@staticmethod
def define_untyped_pipeline(state, name="untyped"):
    """Returns an untyped part of the nopython pipeline"""
    from numba.core.compiler_machinery import PassManager
    from numba.core.untyped_passes import (
        DeadBranchPrune,
        FindLiterallyCalls,
        FixupArgs,
        GenericRewrites,
        InlineClosureLikes,
        InlineInlinables,
        IRProcessing,
        LiteralPropagationSubPipelinePass,
        LiteralUnroll,
        MakeFunctionToJitFunction,
        ReconstructSSA,
        RewriteSemanticConstants,
        TranslateByteCode,
        WithLifting,
    )
    from numba.core.utils import PYVERSION

    pm = PassManager(name)
    if state.func_ir is None:
        pm.add_pass(TranslateByteCode, "analyzing bytecode")
        # Bodo Change: Insert Python 3.10 Bytecode peepholes
        if PYVERSION == (3, 10):
            pm.add_pass(Bodo310ByteCodePass, "Apply Python 3.10 bytecode changes")
        pm.add_pass(FixupArgs, "fix up args")
    pm.add_pass(IRProcessing, "processing IR")
    pm.add_pass(WithLifting, "Handle with contexts")

    # inline closures early in case they are using nonlocal's
    # see issue #6585.
    pm.add_pass(InlineClosureLikes, "inline calls to locally defined closures")

    # pre typing
    if not state.flags.no_rewrites:
        pm.add_pass(RewriteSemanticConstants, "rewrite semantic constants")
        pm.add_pass(DeadBranchPrune, "dead branch pruning")
        pm.add_pass(GenericRewrites, "nopython rewrites")

    # convert any remaining closures into functions
    pm.add_pass(MakeFunctionToJitFunction, "convert make_function into JIT functions")
    # inline functions that have been determined as inlinable and rerun
    # branch pruning, this needs to be run after closures are inlined as
    # the IR repr of a closure masks call sites if an inlinable is called
    # inside a closure
    pm.add_pass(InlineInlinables, "inline inlinable functions")
    if not state.flags.no_rewrites:
        pm.add_pass(DeadBranchPrune, "dead branch pruning")

    pm.add_pass(FindLiterallyCalls, "find literally calls")
    pm.add_pass(LiteralUnroll, "handles literal_unroll")

    if state.flags.enable_ssa:
        pm.add_pass(ReconstructSSA, "ssa")

    pm.add_pass(LiteralPropagationSubPipelinePass, "Literal propagation")

    pm.finalize()
    return pm


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(
        numba.core.compiler.DefaultPassBuilder.define_untyped_pipeline
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "fc5a0665658cc30588a78aca984ac2d323d5d3a45dce538cc62688530c772896"
    ):
        warnings.warn(
            "numba.core.compiler.DefaultPassBuilder.define_untyped_pipeline has changed"
        )
numba.core.compiler.DefaultPassBuilder.define_untyped_pipeline = define_untyped_pipeline

#########  End bytecode changes for Python 3.10  #########


#### BEGIN MONKEY PATCH FOR INT * LIST SUPPORT ####
# TODO: Remove once https://github.com/numba/numba/pull/7848 is merged


def mul_list_generic(self, args, kws):
    a, b = args
    if isinstance(a, types.List) and isinstance(b, types.Integer):
        return signature(a, a, types.intp)
    # Bodo Change: Add typing for Int * List
    elif isinstance(a, types.Integer) and isinstance(b, types.List):
        return signature(b, types.intp, b)


if _check_numba_change:  # pragma: no cover  # pragma: no cover
    lines = inspect.getsource(numba.core.typing.listdecl.MulList.generic)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "95882385a8ffa67aa576e8169b9ee6b3197e0ad3d5def4b47fa65ce8cd0f1575"
    ):  # pragma: no cover
        warnings.warn("numba.core.typing.listdecl.MulList.generic has changed")
numba.core.typing.listdecl.MulList.generic = mul_list_generic

# Bodo Change: Reimplement with indices flipped.
@lower_builtin(operator.mul, types.Integer, types.List)
def list_mul(context, builder, sig, args):
    from llvmlite import ir as lir
    from numba.core.imputils import impl_ret_new_ref
    from numba.cpython.listobj import ListInstance

    if isinstance(sig.args[0], types.List):
        list_idx, int_idx = 0, 1
    else:
        list_idx, int_idx = 1, 0
    src = ListInstance(context, builder, sig.args[list_idx], args[list_idx])
    src_size = src.size

    mult = args[int_idx]
    zero = lir.Constant(mult.type, 0)
    mult = builder.select(cgutils.is_neg_int(builder, mult), zero, mult)
    nitems = builder.mul(mult, src_size)

    dest = ListInstance.allocate(context, builder, sig.return_type, nitems)
    dest.size = nitems

    with cgutils.for_range_slice(builder, zero, nitems, src_size, inc=True) as (
        dest_offset,
        _,
    ):
        with cgutils.for_range(builder, src_size) as loop:
            value = src.getitem(loop.index)
            dest.setitem(builder.add(loop.index, dest_offset), value, incref=True)

    return impl_ret_new_ref(context, builder, sig.return_type, dest.value)


#### END MONKEY PATCH FOR INT * LIST SUPPORT   ####


#### START MONKEY PATCH SUPPORT FOR unifying unknown types ####


def unify_pairs(self, first, second):
    """
    Try to unify the two given types.  A third type is returned,
    or None in case of failure.
    """
    from numba.core.typeconv import Conversion

    if first == second:
        return first

    if first is types.undefined:
        return second
    elif second is types.undefined:
        return first

    # Bodo Change: If either type is unknown, the unified
    # type should also be unknown.
    if first is types.unknown or second is types.unknown:
        return types.unknown

    # Types with special unification rules
    unified = first.unify(self, second)
    if unified is not None:
        return unified

    unified = second.unify(self, first)
    if unified is not None:
        return unified

    # Other types with simple conversion rules
    conv = self.can_convert(fromty=first, toty=second)
    if conv is not None and conv <= Conversion.safe:
        # Can convert from first to second
        return second

    conv = self.can_convert(fromty=second, toty=first)
    if conv is not None and conv <= Conversion.safe:
        # Can convert from second to first
        return first

    if isinstance(first, types.Literal) or isinstance(second, types.Literal):
        first = types.unliteral(first)
        second = types.unliteral(second)
        return self.unify_pairs(first, second)

    # Cannot unify
    return None


if _check_numba_change:  # pragma: no cover
    lines = inspect.getsource(numba.core.typing.context.BaseContext.unify_pairs)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "f0eaf4cfdf1537691de26efd24d7e320f7c3f10d35e9aefe70cb946b3be0008c"
    ):
        warnings.warn("numba.core.typing.context.BaseContext.unify_pairs has changed")

numba.core.typing.context.BaseContext.unify_pairs = unify_pairs

#### END MONKEY PATCH SUPPORT FOR unifying unknown types ####


#### BEGIN MONKEY PATCH FOR REFERENCE COUNTED SET SUPPORT ####


def _native_set_to_python_list(typ, payload, c):
    """
    Create a Python list from a native set's items.
    """
    from llvmlite import ir

    nitems = payload.used
    listobj = c.pyapi.list_new(nitems)
    ok = cgutils.is_not_null(c.builder, listobj)
    with c.builder.if_then(ok, likely=True):
        index = cgutils.alloca_once_value(c.builder, ir.Constant(nitems.type, 0))
        with payload._iterate() as loop:
            i = c.builder.load(index)
            item = loop.entry.key
            # Bodo change: added incref
            c.context.nrt.incref(c.builder, typ.dtype, item)
            itemobj = c.box(typ.dtype, item)
            c.pyapi.list_setitem(listobj, i, itemobj)
            i = c.builder.add(i, ir.Constant(i.type, 1))
            c.builder.store(i, index)
    return ok, listobj


def _lookup(self, item, h, for_insert=False):
    """
    Lookup the *item* with the given hash values in the entries.
    Return a (found, entry index) tuple:
    - If found is true, <entry index> points to the entry containing
      the item.
    - If found is false, <entry index> points to the empty entry that
      the item can be written to (only if *for_insert* is true)
    """
    from llvmlite import ir

    context = self._context
    builder = self._builder
    intp_t = h.type
    mask = self.mask
    dtype = self._ty.dtype
    tyctx = context.typing_context
    fnty = tyctx.resolve_value_type(operator.eq)
    sig = fnty.get_call_type(tyctx, (dtype, dtype), {})
    # Bodo change: removed error message from reference counted items in set
    eqfn = context.get_function(fnty, sig)

    one = ir.Constant(intp_t, 1)
    five = ir.Constant(intp_t, 5)
    # The perturbation value for probing
    perturb = cgutils.alloca_once_value(builder, h)
    # The index of the entry being considered: start with (hash & mask)
    index = cgutils.alloca_once_value(builder, builder.and_(h, mask))
    if for_insert:
        # The index of the first deleted entry in the lookup chain
        free_index_sentinel = mask.type(-1)  # highest unsigned index
        free_index = cgutils.alloca_once_value(builder, free_index_sentinel)
    bb_body = builder.append_basic_block("lookup.body")
    bb_found = builder.append_basic_block("lookup.found")
    bb_not_found = builder.append_basic_block("lookup.not_found")
    bb_end = builder.append_basic_block("lookup.end")

    def check_entry(i):
        """
        Check entry *i* against the value being searched for.
        """
        entry = self.get_entry(i)
        entry_hash = entry.hash
        with builder.if_then(builder.icmp_unsigned("==", h, entry_hash)):
            # Hashes are equal, compare values
            # (note this also ensures the entry is used)
            eq = eqfn(builder, (item, entry.key))
            with builder.if_then(eq):
                builder.branch(bb_found)
        with builder.if_then(
            numba.cpython.setobj.is_hash_empty(context, builder, entry_hash)
        ):
            builder.branch(bb_not_found)
        if for_insert:
            # Memorize the index of the first deleted entry
            with builder.if_then(
                numba.cpython.setobj.is_hash_deleted(context, builder, entry_hash)
            ):
                j = builder.load(free_index)
                j = builder.select(
                    builder.icmp_unsigned("==", j, free_index_sentinel), i, j
                )
                builder.store(j, free_index)

    # First linear probing.  When the number of collisions is small,
    # the lineary probing loop achieves better cache locality and
    # is also slightly cheaper computationally.
    with cgutils.for_range(
        builder, ir.Constant(intp_t, numba.cpython.setobj.LINEAR_PROBES)
    ):
        i = builder.load(index)
        check_entry(i)
        i = builder.add(i, one)
        i = builder.and_(i, mask)
        builder.store(i, index)
    # If not found after linear probing, switch to a non-linear
    # perturbation keyed on the unmasked hash value.
    # XXX how to tell LLVM this branch is unlikely?
    builder.branch(bb_body)
    with builder.goto_block(bb_body):
        i = builder.load(index)
        check_entry(i)
        # Perturb to go to next entry:
        #   perturb >>= 5
        #   i = (i * 5 + 1 + perturb) & mask
        p = builder.load(perturb)
        p = builder.lshr(p, five)
        i = builder.add(one, builder.mul(i, five))
        i = builder.and_(mask, builder.add(i, p))
        builder.store(i, index)
        builder.store(p, perturb)
        # Loop
        builder.branch(bb_body)
    with builder.goto_block(bb_not_found):
        if for_insert:
            # Not found => for insertion, return the index of the first
            # deleted entry (if any), to avoid creating an infinite
            # lookup chain (issue #1913).
            i = builder.load(index)
            j = builder.load(free_index)
            i = builder.select(
                builder.icmp_unsigned("==", j, free_index_sentinel), i, j
            )
            builder.store(i, index)
        builder.branch(bb_end)
    with builder.goto_block(bb_found):
        builder.branch(bb_end)
    builder.position_at_end(bb_end)
    found = builder.phi(ir.IntType(1), "found")
    found.add_incoming(cgutils.true_bit, bb_found)
    found.add_incoming(cgutils.false_bit, bb_not_found)
    return found, builder.load(index)


def _add_entry(self, payload, entry, item, h, do_resize=True):
    context = self._context
    builder = self._builder

    old_hash = entry.hash
    entry.hash = h
    # Bodo change: added incref
    context.nrt.incref(builder, self._ty.dtype, item)
    entry.key = item
    # used++
    used = payload.used
    one = ir.Constant(used.type, 1)
    used = payload.used = builder.add(used, one)
    # fill++ if entry wasn't a deleted one
    with builder.if_then(
        numba.cpython.setobj.is_hash_empty(context, builder, old_hash), likely=True
    ):
        payload.fill = builder.add(payload.fill, one)
    # Grow table if necessary
    if do_resize:
        self.upsize(used)
    self.set_dirty(True)


def _add_key(self, payload, item, h, do_resize=True):
    from llvmlite import ir

    context = self._context
    builder = self._builder
    found, i = payload._lookup(item, h, for_insert=True)
    not_found = builder.not_(found)
    with builder.if_then(not_found):
        # Not found => add it
        entry = payload.get_entry(i)
        old_hash = entry.hash
        entry.hash = h
        # Bodo change: added incref
        context.nrt.incref(builder, self._ty.dtype, item)
        entry.key = item
        # used++
        used = payload.used
        one = ir.Constant(used.type, 1)
        used = payload.used = builder.add(used, one)
        # fill++ if entry wasn't a deleted one
        with builder.if_then(
            numba.cpython.setobj.is_hash_empty(context, builder, old_hash), likely=True
        ):
            payload.fill = builder.add(payload.fill, one)
        # Grow table if necessary
        if do_resize:
            self.upsize(used)
        self.set_dirty(True)


def _remove_entry(self, payload, entry, do_resize=True):
    from llvmlite import ir

    # Mark entry deleted
    entry.hash = ir.Constant(entry.hash.type, numba.cpython.setobj.DELETED)
    # Bodo change: added decref
    self._context.nrt.decref(self._builder, self._ty.dtype, entry.key)
    # used--
    used = payload.used
    one = ir.Constant(used.type, 1)
    used = payload.used = self._builder.sub(used, one)
    # Shrink table if necessary
    if do_resize:
        self.downsize(used)
    self.set_dirty(True)


def pop(self):
    context = self._context
    builder = self._builder
    lty = context.get_value_type(self._ty.dtype)
    key = cgutils.alloca_once(builder, lty)
    payload = self.payload
    with payload._next_entry() as entry:
        builder.store(entry.key, key)
        # Bodo change: added incref
        # incref since the value is returned but _remove_entry() decrefs
        context.nrt.incref(builder, self._ty.dtype, entry.key)
        self._remove_entry(payload, entry)

    return builder.load(key)


def _resize(self, payload, nentries, errmsg):
    """
    Resize the payload to the given number of entries.
    CAUTION: *nentries* must be a power of 2!
    """
    context = self._context
    builder = self._builder
    # Allocate new entries
    old_payload = payload
    ok = self._allocate_payload(nentries, realloc=True)
    with builder.if_then(builder.not_(ok), likely=False):
        context.call_conv.return_user_exc(builder, MemoryError, (errmsg,))

    # Bodo change: added decref
    # Re-insert old entries
    # Decref to balance incref from `_add_key`
    payload = self.payload
    with old_payload._iterate() as loop:
        entry = loop.entry
        self._add_key(payload, entry.key, entry.hash, do_resize=False)
        context.nrt.decref(builder, self._ty.dtype, entry.key)

    self._free_payload(old_payload.ptr)


def _replace_payload(self, nentries):
    """
    Replace the payload with a new empty payload with the given number
    of entries.
    CAUTION: *nentries* must be a power of 2!
    """
    context = self._context
    builder = self._builder

    # Bodo change: added decref
    # decref all of the previous entries
    with self.payload._iterate() as loop:
        entry = loop.entry
        context.nrt.decref(builder, self._ty.dtype, entry.key)

    # Free old payload
    self._free_payload(self.payload.ptr)

    ok = self._allocate_payload(nentries, realloc=True)
    with builder.if_then(builder.not_(ok), likely=False):
        context.call_conv.return_user_exc(
            builder, MemoryError, ("cannot reallocate set",)
        )


def _allocate_payload(self, nentries, realloc=False):
    """
    Allocate and initialize payload for the given number of entries.
    If *realloc* is True, the existing meminfo is reused.
    CAUTION: *nentries* must be a power of 2!
    """
    from llvmlite import ir

    context = self._context
    builder = self._builder
    ok = cgutils.alloca_once_value(builder, cgutils.true_bit)
    intp_t = context.get_value_type(types.intp)
    zero = ir.Constant(intp_t, 0)
    one = ir.Constant(intp_t, 1)
    payload_type = context.get_data_type(types.SetPayload(self._ty))
    payload_size = context.get_abi_sizeof(payload_type)
    entry_size = self._entrysize
    # Account for the fact that the payload struct already contains an entry
    payload_size -= entry_size
    # Total allocation size = <payload header size> + nentries * entry_size
    allocsize, ovf = cgutils.muladd_with_overflow(
        builder,
        nentries,
        ir.Constant(intp_t, entry_size),
        ir.Constant(intp_t, payload_size),
    )
    with builder.if_then(ovf, likely=False):
        builder.store(cgutils.false_bit, ok)
    with builder.if_then(builder.load(ok), likely=True):
        if realloc:
            meminfo = self._set.meminfo
            ptr = context.nrt.meminfo_varsize_alloc(builder, meminfo, size=allocsize)
            alloc_ok = cgutils.is_null(builder, ptr)
        else:
            # Bodo change: added destructor
            # create destructor to be called upon set destruction
            dtor = _imp_dtor(context, builder.module, self._ty)
            meminfo = context.nrt.meminfo_new_varsize_dtor(
                builder, allocsize, builder.bitcast(dtor, cgutils.voidptr_t)
            )
            alloc_ok = cgutils.is_null(builder, meminfo)

        # Bodo change: used `alloc_ok`
        with builder.if_else(alloc_ok, likely=False) as (if_error, if_ok):
            with if_error:
                builder.store(cgutils.false_bit, ok)
            with if_ok:
                if not realloc:
                    self._set.meminfo = meminfo
                    self._set.parent = context.get_constant_null(types.pyobject)
                payload = self.payload
                # Initialize entries to 0xff (EMPTY)
                cgutils.memset(builder, payload.ptr, allocsize, 0xFF)
                payload.used = zero
                payload.fill = zero
                payload.finger = zero
                new_mask = builder.sub(nentries, one)
                payload.mask = new_mask
    return builder.load(ok)


def _copy_payload(self, src_payload):
    """
    Raw-copy the given payload into self.
    """
    from llvmlite import ir

    context = self._context
    builder = self._builder
    ok = cgutils.alloca_once_value(builder, cgutils.true_bit)
    intp_t = context.get_value_type(types.intp)
    zero = ir.Constant(intp_t, 0)
    one = ir.Constant(intp_t, 1)
    payload_type = context.get_data_type(types.SetPayload(self._ty))
    payload_size = context.get_abi_sizeof(payload_type)
    entry_size = self._entrysize
    # Account for the fact that the payload struct already contains an entry
    payload_size -= entry_size
    mask = src_payload.mask
    nentries = builder.add(one, mask)
    # Total allocation size = <payload header size> + nentries * entry_size
    # (note there can't be any overflow since we're reusing an existing
    #  payload's parameters)
    allocsize = builder.add(
        ir.Constant(intp_t, payload_size),
        builder.mul(ir.Constant(intp_t, entry_size), nentries),
    )

    with builder.if_then(builder.load(ok), likely=True):
        # Bodo change: added destructor
        # create destructor for new meminfo
        dtor = _imp_dtor(context, builder.module, self._ty)
        meminfo = context.nrt.meminfo_new_varsize_dtor(
            builder, allocsize, builder.bitcast(dtor, cgutils.voidptr_t)
        )
        alloc_ok = cgutils.is_null(builder, meminfo)

        # Bodo change: used `alloc_ok`
        with builder.if_else(alloc_ok, likely=False) as (if_error, if_ok):
            with if_error:
                builder.store(cgutils.false_bit, ok)
            with if_ok:
                self._set.meminfo = meminfo
                payload = self.payload
                payload.used = src_payload.used
                payload.fill = src_payload.fill
                payload.finger = zero
                payload.mask = mask

                # instead of using `_add_key` for every entry, since the
                # size of the new set is the same, we can just copy the
                # data directly without having to re-compute the hash
                cgutils.raw_memcpy(
                    builder, payload.entries, src_payload.entries, nentries, entry_size
                )
                # Bodo change: added increfs
                # increment the refcounts to simulate `_add_key` for each
                # element
                with src_payload._iterate() as loop:
                    context.nrt.incref(builder, self._ty.dtype, loop.entry.key)

    return builder.load(ok)


# Bodo change: added `_imp_dtor`
def _imp_dtor(context, module, set_type):
    """Define the dtor for set"""
    from llvmlite import ir

    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    # create a dtor function that takes (void* set, size_t size, void* dtor_info)
    fnty = ir.FunctionType(
        ir.VoidType(),
        [llvoidptr, llsize, llvoidptr],
    )
    # create type-specific name
    fname = f"_numba_set_dtor_{set_type}"

    fn = cgutils.get_or_insert_function(module, fnty, name=fname)

    if fn.is_declaration:
        # Set linkage
        fn.linkage = "linkonce_odr"
        # Define
        builder = ir.IRBuilder(fn.append_basic_block())
        set_ptr = builder.bitcast(fn.args[0], cgutils.voidptr_t.as_pointer())
        payload = numba.cpython.setobj._SetPayload(context, builder, set_type, set_ptr)
        with payload._iterate() as loop:
            entry = loop.entry
            context.nrt.decref(builder, set_type.dtype, entry.key)
        builder.ret_void()

    return fn


@lower_builtin(set, types.IterableType)
def set_constructor(context, builder, sig, args):
    set_type = sig.return_type
    (items_type,) = sig.args
    (items,) = args

    # Bodo change: added decref
    # If the argument has a len(), preallocate the set so as to
    # avoid resizes.
    # both `add` and `for_iter` incref each item in the set, so manually decref
    # the items to avoid a leak from the double incref
    n = numba.core.imputils.call_len(context, builder, items_type, items)
    inst = numba.cpython.setobj.SetInstance.allocate(context, builder, set_type, n)
    with numba.core.imputils.for_iter(context, builder, items_type, items) as loop:
        inst.add(loop.value)
        context.nrt.decref(builder, set_type.dtype, loop.value)

    return numba.core.imputils.impl_ret_new_ref(context, builder, set_type, inst.value)


@lower_builtin("set.update", types.Set, types.IterableType)
def set_update(context, builder, sig, args):
    inst = numba.cpython.setobj.SetInstance(context, builder, sig.args[0], args[0])
    items_type = sig.args[1]
    items = args[1]
    # If the argument has a len(), assume there are few collisions and
    # presize to len(set) + len(items)
    n = numba.core.imputils.call_len(context, builder, items_type, items)
    if n is not None:
        new_size = builder.add(inst.payload.used, n)
        inst.upsize(new_size)
    with numba.core.imputils.for_iter(context, builder, items_type, items) as loop:
        # make sure that the items being added are of the same dtype as the
        # set instance
        casted = context.cast(builder, loop.value, items_type.dtype, inst.dtype)
        inst.add(casted)
        # Bodo change: added decref
        # decref each item to counter balance the double incref from `add` +
        # `for_iter`
        context.nrt.decref(builder, items_type.dtype, loop.value)

    if n is not None:
        # If we pre-grew the set, downsize in case there were many collisions
        inst.downsize(inst.payload.used)
    return context.get_dummy_value()


if _check_numba_change:  # pragma: no cover
    for name, orig, hash in (
        (
            "numba.core.boxing._native_set_to_python_list",
            numba.core.boxing._native_set_to_python_list,
            "b47f3d5e582c05d80899ee73e1c009a7e5121e7a660d42cb518bb86933f3c06f",
        ),
        (
            "numba.cpython.setobj._SetPayload._lookup",
            numba.cpython.setobj._SetPayload._lookup,
            "c797b5399d7b227fe4eea3a058b3d3103f59345699388afb125ae47124bee395",
        ),
        (
            "numba.cpython.setobj.SetInstance._add_entry",
            numba.cpython.setobj.SetInstance._add_entry,
            "c5ed28a5fdb453f242e41907cb792b66da2df63282c17abe0b68fc46782a7f94",
        ),
        (
            "numba.cpython.setobj.SetInstance._add_key",
            numba.cpython.setobj.SetInstance._add_key,
            "324d6172638d02a361cfa0ca7f86e241e5a56a008d4ab581a305f9ae5ea4a75f",
        ),
        (
            "numba.cpython.setobj.SetInstance._remove_entry",
            numba.cpython.setobj.SetInstance._remove_entry,
            "2c441b00daac61976e673c0e738e8e76982669bd2851951890dd40526fa14da1",
        ),
        (
            "numba.cpython.setobj.SetInstance.pop",
            numba.cpython.setobj.SetInstance.pop,
            "1a7b7464cbe0577f2a38f3af9acfef6d4d25d049b1e216157275fbadaab41d1b",
        ),
        (
            "numba.cpython.setobj.SetInstance._resize",
            numba.cpython.setobj.SetInstance._resize,
            "5ca5c2ba4f8c4bf546fde106b9c2656d4b22a16d16e163fb64c5d85ea4d88746",
        ),
        (
            "numba.cpython.setobj.SetInstance._replace_payload",
            numba.cpython.setobj.SetInstance._replace_payload,
            "ada75a6c85828bff69c8469538c1979801f560a43fb726221a9c21bf208ae78d",
        ),
        (
            "numba.cpython.setobj.SetInstance._allocate_payload",
            numba.cpython.setobj.SetInstance._allocate_payload,
            "2e80c419df43ebc71075b4f97fc1701c10dbc576aed248845e176b8d5829e61b",
        ),
        (
            "numba.cpython.setobj.SetInstance._copy_payload",
            numba.cpython.setobj.SetInstance._copy_payload,
            "0885ac36e1eb5a0a0fc4f5d91e54b2102b69e536091fed9f2610a71d225193ec",
        ),
        (
            "numba.cpython.setobj.set_constructor",
            numba.cpython.setobj.set_constructor,
            "3d521a60c3b8eaf70aa0f7267427475dfddd8f5e5053b5bfe309bb5f1891b0ce",
        ),
        (
            "numba.cpython.setobj.set_update",
            numba.cpython.setobj.set_update,
            "965c4f7f7abcea5cbe0491b602e6d4bcb1800fa1ec39b1ffccf07e1bc56051c3",
        ),
    ):
        lines = inspect.getsource(orig)
        if hashlib.sha256(lines.encode()).hexdigest() != hash:
            warnings.warn(f"{name} has changed")
        orig = new


numba.core.boxing._native_set_to_python_list = _native_set_to_python_list
numba.cpython.setobj._SetPayload._lookup = _lookup
numba.cpython.setobj.SetInstance._add_entry = _add_entry
numba.cpython.setobj.SetInstance._add_key = _add_key
numba.cpython.setobj.SetInstance._remove_entry = _remove_entry
numba.cpython.setobj.SetInstance.pop = pop
numba.cpython.setobj.SetInstance._resize = _resize
numba.cpython.setobj.SetInstance._replace_payload = _replace_payload
numba.cpython.setobj.SetInstance._allocate_payload = _allocate_payload
numba.cpython.setobj.SetInstance._copy_payload = _copy_payload

#### END MONKEY PATCH FOR REFERENCE COUNTED SET SUPPORT ####

#### BEGIN MONKEY PATCH FOR METADATA CACHING SUPPORT ####


def _reduce(self):
    """
    Reduce a CompileResult to picklable components.
    """
    libdata = self.library.serialize_using_object_code()
    # Make it (un)picklable efficiently
    typeann = str(self.type_annotation)
    fndesc = self.fndesc
    # Those don't need to be pickled and may fail
    fndesc.typemap = fndesc.calltypes = None
    # Include all referenced environments
    referenced_envs = self._find_referenced_environments()
    # Bodo change: filter metadata
    bodo_metadata = {
        key: value
        for key, value in self.metadata.items()
        if ("distributed" in key or "replicated" in key)
        and key != "distributed_diagnostics"  # TODO: [BE-2617] remove this
    }
    return (
        libdata,
        self.fndesc,
        self.environment,
        self.signature,
        self.objectmode,
        self.lifted,
        typeann,
        # Bodo change: add metadata
        bodo_metadata,
        self.reload_init,
        tuple(referenced_envs),
    )


@classmethod
def _rebuild(
    cls,
    target_context,
    libdata,
    fndesc,
    env,
    signature,
    objectmode,
    lifted,
    typeann,
    # Bodo change: add metadata
    metadata,
    reload_init,
    referenced_envs,
):
    if reload_init:
        # Re-run all
        for fn in reload_init:
            fn()

    library = target_context.codegen().unserialize_library(libdata)
    cfunc = target_context.get_executable(library, fndesc, env)
    cr = cls(
        target_context=target_context,
        typing_context=target_context.typing_context,
        library=library,
        environment=env,
        entry_point=cfunc,
        fndesc=fndesc,
        type_annotation=typeann,
        signature=signature,
        objectmode=objectmode,
        lifted=lifted,
        typing_error=None,
        call_helper=None,
        # Bodo change: add metadata
        metadata=metadata,
        reload_init=reload_init,
        referenced_envs=referenced_envs,
    )

    # Load Environments
    for env in referenced_envs:
        library.codegen.set_env(env.env_name, env)

    return cr


if _check_numba_change:  # pragma: no cover
    for name, orig, hash in (
        (
            "numba.core.compiler.CompileResult._reduce",
            numba.core.compiler.CompileResult._reduce,
            "5f86eacfa5202c202b3dc200f1a7a9b6d3f9d1ec16d43a52cb2d580c34fbfa82",
        ),
        (
            "numba.core.compiler.CompileResult._rebuild",
            numba.core.compiler.CompileResult._rebuild,
            "44fa9dc2255883ab49195d18c3cca8c0ad715d0dd02033bd7e2376152edc4e84",
        ),
    ):
        lines = inspect.getsource(orig)
        if hashlib.sha256(lines.encode()).hexdigest() != hash:
            warnings.warn(f"{name} has changed")
        orig = new

numba.core.compiler.CompileResult._reduce = _reduce
numba.core.compiler.CompileResult._rebuild = _rebuild

#### END MONKEY PATCH FOR METADATA CACHING SUPPORT ####

#### BEGIN MONKEY PATCH FOR CACHING TO SPECIFIC DIRECTORY FROM IPYTHON NOTEBOOKS ####


def _get_cache_path(self):
    # _UserProvidedCacheLocator uses os.path.join(config.CACHE_DIR, cache_subpath), where cache_subpath
    # is derived from the python file being cached
    return numba.config.CACHE_DIR


if _check_numba_change:  # pragma: no cover
    if os.environ.get("BODO_PLATFORM_CACHE_LOCATION") is not None:
        numba.core.caching._IPythonCacheLocator.get_cache_path = _get_cache_path

#### END MONKEY PATCH FOR CACHING TO SPECIFIC DIRECTORY FROM IPYTHON NOTEBOOKS ####
