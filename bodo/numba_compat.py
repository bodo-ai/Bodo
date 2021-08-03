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
import numba.np.linalg
from numba.core import analysis, cgutils, errors, ir, ir_utils, types
from numba.core.errors import ForceLiteralArg, LiteralTypingError, TypingError
from numba.core.ir_utils import (
    GuardException,
    _create_function_from_code_obj,
    analysis,
    build_definitions,
    find_callname,
    guard,
    has_no_side_effect,
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
    _OverloadMethodTemplate,
    infer_global,
    signature,
)
from numba.core.typing.typeof import Purpose, typeof
from numba.experimental.jitclass import base as jitclass_base
from numba.experimental.jitclass import decorators as jitclass_decorators
from numba.extending import NativeValue, lower_builtin, typeof_impl
from numba.parfors.parfor import get_expr_args

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
    # XXX make this a dedicated Pipeline?
    func_id = numba.core.bytecode.FunctionIdentity.from_function(func)
    interp = numba.core.interpreter.Interpreter(func_id)
    bc = numba.core.bytecode.ByteCode(func_id=func_id)
    func_ir = interp.interpret(bc)
    if inline_closures:
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
        inline_pass = numba.core.inline_closurecall.InlineClosureCallPass(
            func_ir, numba.core.cpu.ParallelOptions(False), {}, False
        )
        inline_pass.run()
    post_proc = numba.core.postproc.PostProcessor(func_ir)
    post_proc.run(emit_dels)
    return func_ir


if _check_numba_change:
    # make sure run_frontend hasn't changed before replacing it
    lines = inspect.getsource(numba.core.compiler.run_frontend)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "53e6495bcf751f5fdd91ce6f9319cd7e06df8c3e2919fdf3f9fda976ff4c83ea"
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


if _check_numba_change:
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
        state.func_ir, state.typingctx, state.typemap, state.calltypes
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


if _check_numba_change:
    # make sure find_potential_aliases hasn't changed before replacing it
    lines = inspect.getsource(ir_utils.find_potential_aliases)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "2b17b56512a6b9c95e7c6c072bb2e16f681fe2e8e4b8cb7b9fc7ac83133361a1"
    ):  # pragma: no cover
        warnings.warn("ir_utils.find_potential_aliases has changed")


ir_utils.find_potential_aliases = find_potential_aliases


if _check_numba_change:
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
                    if typemap and isinstance(typemap.get(lhs, None), types.Function):
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
        block.body = new_body


ir_utils.dead_code_elimination = mini_dce
numba.core.typed_passes.dead_code_elimination = mini_dce
numba.core.inline_closurecall.dead_code_elimination = mini_dce


# replace Numba's overload/overload_method handling functions to support a new option
# called 'no_unliteral', which avoids a second run of overload with literal types
# converted to non-literal versions. This solves hiding errors such as #889
# TODO: remove after Numba's #5411 is resolved
_overload_default_jit_options = {"no_cpython_wrapper": True}


# change: added no_unliteral argument
def overload(
    func,
    jit_options={},
    strict=True,
    inline="never",
    prefer_literal=False,
    no_unliteral=False,
):
    from numba.core.typing.templates import (
        infer,
        infer_global,
        make_overload_template,
    )

    # set default options
    opts = _overload_default_jit_options.copy()
    opts.update(jit_options)  # let user options override

    def decorate(overload_func):
        # change: added no_unliteral argument
        template = make_overload_template(
            func, overload_func, opts, strict, inline, prefer_literal, no_unliteral
        )
        infer(template)
        if callable(func):
            infer_global(func, types.Function(template))
        return overload_func

    return decorate


if _check_numba_change:
    # make sure overload hasn't changed before replacing it
    lines = inspect.getsource(numba.core.extending.overload)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "9a2e592c984bd1d2df462bc3f2724a155962f85a30dc468d67acbd78cb023cc5"
    ):  # pragma: no cover
        warnings.warn("numba.core.extending.overload has changed")


numba.core.extending.overload = overload
numba.extending.overload = overload


def overload_method(typ, attr, **kwargs):
    from numba.core.typing.templates import (
        infer_getattr,
        make_overload_method_template,
    )

    def decorate(overload_func):
        template = make_overload_method_template(
            typ,
            attr,
            overload_func,
            inline=kwargs.get("inline", "never"),
            prefer_literal=kwargs.get("prefer_literal", False),
            # change: added no_unliteral argument
            no_unliteral=kwargs.get("no_unliteral", False),
        )
        infer_getattr(template)
        overload(overload_func, **kwargs)(overload_func)
        return overload_func

    return decorate


if _check_numba_change:
    # make sure overload_method hasn't changed before replacing it
    lines = inspect.getsource(numba.core.extending.overload_method)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "0d2663f1499836c32413f6a4dd8e4c1a407453320034b81b343e38fc34cf0658"
    ):  # pragma: no cover
        warnings.warn("numba.core.extending.overload_method has changed")


numba.core.extending.overload_method = overload_method
numba.extending.overload_method = overload_method


from numba.core.cpu_options import InlineOptions


# change: added no_unliteral argument
def make_overload_template(
    func,
    overload_func,
    jit_options,
    strict,
    inline,
    prefer_literal=False,
    no_unliteral=False,
):
    """
    Make a template class for function *func* overloaded by *overload_func*.
    Compiler options are passed as a dictionary to *jit_options*.
    """
    func_name = getattr(func, "__name__", str(func))
    name = "OverloadTemplate_%s" % (func_name,)
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
    )
    return type(base)(name, (base,), dct)


if _check_numba_change:
    # make sure make_overload_template hasn't changed before replacing it
    lines = inspect.getsource(numba.core.typing.templates.make_overload_template)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "1d304a414a1bfb2d7185ddddabcf637790ef2357b4c7d90035970b4f60e2d058"
    ):  # pragma: no cover
        warnings.warn("numba.core.typing.templates.make_overload_template has changed")


numba.core.typing.templates.make_overload_template = make_overload_template


def _resolve(self, typ, attr):
    if self._attr != attr:
        return None

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


if _check_numba_change:
    # make sure _resolve hasn't changed before replacing it
    lines = inspect.getsource(
        numba.core.typing.templates._OverloadMethodTemplate._resolve
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "ccfe9f8dec20f58cf61f95420e25abe5419faf2cfffde78f39118fe6f91949e3"
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
    no_unliteral=False,
    base=_OverloadAttributeTemplate,
):
    """
    Make a template class for attribute *attr* of *typ* overloaded by
    *overload_func*.
    """
    assert isinstance(typ, types.Type) or issubclass(typ, types.Type)
    name = "OverloadAttributeTemplate_%s_%s" % (typ, attr)
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
    )
    obj = type(base)(name, (base,), dct)
    return obj


if _check_numba_change:
    # make sure make_overload_attribute_template hasn't changed before replacing it
    lines = inspect.getsource(
        numba.core.typing.templates.make_overload_attribute_template
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "fced9b9ca8b1f94d3f6b5fd2377acc544df3be01dfd4adf7044d6e27473a357a"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.typing.templates.make_overload_attribute_template has changed"
        )


numba.core.typing.templates.make_overload_attribute_template = (
    make_overload_attribute_template
)


# change: added no_unliteral argument
def make_overload_method_template(
    typ, attr, overload_func, inline, prefer_literal=False, no_unliteral=False
):
    """
    Make a template class for method *attr* of *typ* overloaded by
    *overload_func*.
    """
    return make_overload_attribute_template(
        # Bodo change: added no_unliteral argument
        typ,
        attr,
        overload_func,
        inline=inline,
        no_unliteral=no_unliteral,
        base=_OverloadMethodTemplate,
        prefer_literal=prefer_literal,
    )


if _check_numba_change:
    # make sure make_overload_method_template hasn't changed before replacing it
    lines = inspect.getsource(numba.core.typing.templates.make_overload_method_template)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "09157f571dec522776accc54e9ec9261300100cac107c0cc94b7d17fb51238a1"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.typing.templates.make_overload_method_template has changed"
        )


numba.core.typing.templates.make_overload_method_template = (
    make_overload_method_template
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
            self.context, ir, folded_args, None
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
        self._compiled_overloads[sig.args] = disp_type.get_overload(sig)
    return sig


# make sure generic() hasn't changed before replacing it
lines = inspect.getsource(numba.core.typing.templates._OverloadFunctionTemplate.generic)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "1205804c405eecc4e15e0c9e3695f6d4ce0957a3c60907cdcb629c694cd094ce"
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


if _check_numba_change:
    # make sure bound_function hasn't changed before replacing it
    lines = inspect.getsource(numba.core.typing.templates.bound_function)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "a2feefe64eae6a15c56affc47bf0c1d04461f9566913442d539452b397103322"
    ):  # pragma: no cover
        warnings.warn("numba.core.typing.templates.bound_function has changed")


numba.core.typing.templates.bound_function = bound_function


def get_call_type(self, context, args, kws):
    prefer_lit = [True, False]  # old behavior preferring literal
    prefer_not = [False, True]  # new behavior preferring non-literal
    failures = _ResolutionFailures(context, self, args, kws, depth=self._depth)
    self._depth += 1
    for temp_cls in self.templates:
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

    if len(failures) == 0:
        raise AssertionError(
            "Internal Error. "
            "Function resolution ended with no failures "
            "or successful signature"
        )
    failures.raise_error()


if _check_numba_change:
    # make sure get_call_type hasn't changed before replacing it
    lines = inspect.getsource(numba.core.types.functions.BaseFunction.get_call_type)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "eea3bba0b35522f006f451309e9155f1fbfd94448060e2c9763df8a37d105880"
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
            msg = "Compilation error for "
            # TODO add other data types
            if isinstance(self.this, bodo.hiframes.pd_dataframe_ext.DataFrameType):
                msg += "DataFrame."
            elif isinstance(self.this, bodo.hiframes.pd_series_ext.SeriesType):
                msg += "Series."
            msg += f"{self.typing_key[1]}().{bodo_typing_error_info}"
            raise errors.TypingError(msg)
    return out


if _check_numba_change:
    # make sure get_call_type hasn't changed before replacing it
    lines = inspect.getsource(numba.core.types.functions.BoundFunction.get_call_type)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "5427d7ba522b97a4e34745587365b1eacb7b9641229649a02737f944e150bfba"
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
    from llvmlite.llvmpy.core import Type

    fnty = Type.function(self.pyobj, [self.cstring, self.py_ssize_t])
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
                        (i, "cannot determine Numba type of value %r" % (val,))
                    )
        if failed_args:
            # Patch error message to ease debugging
            msg = str(e).rstrip() + (
                "\n\nThis error may have been caused by the following argument(s):\n%s\n"
                % "\n".join("- argument %d: %s" % (i, err) for i, err in failed_args)
            )
            e.patch_message(msg)

        # In user mode if error comes from numba lowering, suppress stack.
        # Only if it has not been suppressed earlier (because of TypingError in Bodo).
        if not numba.core.config.DEVELOPER_MODE:
            # If error_info is already there, that means Bodo already suppressed stack.
            if bodo_typing_error_info not in e.msg:
                # This is a Numba error
                numba_deny_list = [
                    "Failed in nopython mode pipeline",
                    "Failed in bodo mode pipeline",
                    "numba",
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

if _check_numba_change:
    lines = inspect.getsource(numba.core.dispatcher._DispatcherBase._compile_for_args)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "79466480839c6e16cf437dc054937deb639c85664df1fef673e8f34dfe9d41b6"
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


def compile(self, sig):
    import numba.core.event as ev
    from numba.core import sigutils
    from numba.core.compiler_lock import global_compiler_lock

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
            self._cache.save_overload(sig, cres)
            return cres.entry_point


if _check_numba_change:
    lines = inspect.getsource(numba.core.dispatcher.Dispatcher.compile)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "bab1d95438fe39d3e26e0ac0989ae3a3b54439ee0eda8507a693ea6ca4a4917b"
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


if _check_numba_change:
    lines = inspect.getsource(numba.core.codegen.CodeLibrary._get_module_for_linking)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "56dde0e0555b5ec85b93b97c81821bce60784515a1fbf99e4542e92d02ff0a73"
    ):  # pragma: no cover
        warnings.warn(
            "numba.core.codegen.CodeLibrary._get_module_for_linking has changed"
        )

numba.core.codegen.CodeLibrary._get_module_for_linking = _get_module_for_linking


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
                numba.core.typeinfer._logger.debug("captured error", exc_info=e)
                msg = (
                    "Internal error at {con}.\n"
                    "{err}\nEnable logging at debug level for details."
                )
                new_exc = numba.core.errors.TypingError(
                    msg.format(con=constraint, err=str(e)),
                    loc=constraint.loc,
                    highlighting=False,
                )
                errors.append(numba.core.utils.chain_exception(new_exc, e))
    return errors


if _check_numba_change:
    lines = inspect.getsource(numba.core.typeinfer.ConstraintNetwork.propagate)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "2c204df4d8c58da7c86e0abbab48a7a7863ee3cbe8d2ba89f617d4f580b622e9"
    ):  # pragma: no cover
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


if _check_numba_change:
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
            if lhs.name not in lives and has_no_side_effect(
                rhs, lives_n_aliases, call_table
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


# fix Numba's max label global bug
def ParforPassStates__init__(
    self,
    func_ir,
    typemap,
    calltypes,
    return_type,
    typingctx,
    options,
    flags,
    metadata,
    diagnostics=numba.parfors.parfor.ParforDiagnostics(),
):
    self.func_ir = func_ir
    self.typemap = typemap
    self.calltypes = calltypes
    self.typingctx = typingctx
    self.return_type = return_type
    self.options = options
    self.diagnostics = diagnostics
    self.swapped_fns = diagnostics.replaced_fns
    self.fusion_info = diagnostics.fusion_info
    self.nested_fusion_info = diagnostics.nested_fusion_info

    self.array_analysis = numba.parfors.array_analysis.ArrayAnalysis(
        self.typingctx,
        self.func_ir,
        self.typemap,
        self.calltypes,
    )

    # bodo change: make sure _max_label is always maximum
    ir_utils._max_label = max(ir_utils._max_label, max(func_ir.blocks.keys()))
    self.flags = flags
    self.metadata = metadata
    if "parfors" not in metadata:
        metadata["parfors"] = {}


if _check_numba_change:
    lines = inspect.getsource(numba.parfors.parfor.ParforPassStates.__init__)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "86614f7cf5b0ce442ac51d1ade9bc42fd5c238cae723c3b0ed3d8e4d9a33d7fb"
    ):  # pragma: no cover
        warnings.warn("numba.parfors.parfor.ParforPassStates.__init__ has changed")

numba.parfors.parfor.ParforPassStates.__init__ = ParforPassStates__init__


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
    try:
        return literal(value)
    except LiteralTypingError:
        return


if _check_numba_change:
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


if _check_numba_change:
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


if _check_numba_change:
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
        "https://docs.bodo.ai/latest/source/pandas.html#user-defined-functions-udfs"
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
                glbls[func_name] = numba.njit(val)
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
            glbls[func_name] = numba.njit(nested_func)
            freevars.append(func_name)
        else:
            raise bodo.utils.typing.BodoError(msg.format(x), loc=code_obj.loc)

    func_env = "\n".join(["  c_%d = %s" % (i, x) for i, x in enumerate(freevars)])
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


if _check_numba_change:
    lines = inspect.getsource(numba.core.ir_utils.convert_code_obj_to_function)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "b6b51a980f1952532f128fc20653b836a3d45ceb93add91fa14acd54901444d7"
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
                msg = "Failed in %s mode pipeline (step: %s)" % (
                    self.pipeline_name,
                    pass_desc,
                )
                patched_exception = self._patch_error(msg, e)
                raise patched_exception
            else:
                # Remove `Failed in ... pipeline` message
                raise e


if _check_numba_change:
    lines = inspect.getsource(numba.core.compiler_machinery.PassManager.run)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "5d21271317cfa1bdcec1cc71973d80df0ffd7126c4608eeef4ad676bbff8f0d3"
    ):  # pragma: no cover
        warnings.warn("numba.core.compiler_machinery.PassManager.run has changed")

numba.core.compiler_machinery.PassManager.run = passmanager_run


if _check_numba_change:
    lines = inspect.getsource(numba.np.ufunc.parallel._launch_threads)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "3d7e5889ad7dcd2b1ff0389cf37df400855e0b0b25b956927073b49015298736"
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

    def lookup(var, varonly=True):
        val = defs.get(var.name, None)
        if isinstance(val, ir.Var):
            return lookup(val)
        else:
            return var if (varonly or val is None) else val

    name = reduction_node.name
    unversioned_name = reduction_node.unversioned_name
    for i, stmt in enumerate(nodes):
        lhs = stmt.target
        rhs = stmt.value
        defs[lhs.name] = rhs
        if isinstance(rhs, ir.Var) and rhs.name in defs:
            rhs = lookup(rhs)
        if isinstance(rhs, ir.Expr):
            in_vars = set(lookup(v, True).name for v in rhs.list_vars())
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
                args = [(x.name, lookup(x, True)) for x in get_expr_args(rhs)]
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


if _check_numba_change:
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


if _check_numba_change:
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


if _check_numba_change:
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
            writes.update({v.name for v in stmt.out_key_arrs})
            writes.update({v.name for v in stmt.df_out_vars.values()})
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
            ("set_bit_to_arr", "bodo.libs.int_arr_ext"),
        ):
            writes.add(stmt.value.args[0].name)
    return writes


if _check_numba_change:
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


if _check_numba_change:
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


if _check_numba_change:
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


if _check_numba_change:
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


if _check_numba_change:
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
            return register_class_type(
                cls, spec, types.ClassType, jitclass_base.ClassBuilder, **options
            )

    if cls_or_spec is None:
        return wrap
    else:
        return wrap(cls_or_spec)


if _check_numba_change:
    lines = inspect.getsource(jitclass_decorators.jitclass)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "f6ad843b4d553d18f6f0028fa231e38b7861f23533d7f3a8274a18fc17423d9e"
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


if _check_numba_change:
    lines = inspect.getsource(numba.core.typeinfer.CallConstraint.resolve)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "bef85735a781f8ce98211f23227052e3abb12cb9b8120dc2a59840421af8595b"
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


if _check_numba_change:
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


if _check_numba_change:
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
        {**self.file_infos, **other.file_infos},
    )


if _check_numba_change:
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


if _check_numba_change:
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
            assert isinstance(v_const, types.Type)
            typeanns[k] = v_const
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


if _check_numba_change:
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
    from llvmlite.llvmpy.core import Type

    cstr = self.context.insert_const_string(self.module, attr)
    fnty = Type.function(Type.int(), [self.pyobj, self.cstring])
    fn = self._get_function(fnty, name="PyObject_HasAttrString")
    return self.builder.call(fn, [obj, cstr])


numba.core.pythonapi.PythonAPI.object_hasattr_string = object_hasattr_string


# Support unboxing regular dictionaries as Numba's typed dictionary
# TODO(ehsan): move to Numba
# TODO(ehsan): reflection is not supported. Throw warning if dict is modified?
@typeof_impl.register(dict)
def _typeof_dict(val, c):
    if len(val) == 0:
        raise ValueError("Cannot type empty dict")
    k, v = next(iter(val.items()))
    key_type = typeof_impl(k, c)
    value_type = typeof_impl(v, c)
    if key_type is None or value_type is None:
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

            def convert(mi, typ):
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


if _check_numba_change:
    lines = inspect.getsource(
        numba.core.pythonapi._unboxers.functions[numba.core.types.DictType]
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "5f6f183b94dc57838538c668a54c2476576c85d8553843f3219f5162c61e7816"
    ):  # pragma: no cover
        warnings.warn("unbox_dicttype has changed")

numba.core.pythonapi._unboxers.functions[types.DictType] = unbox_dicttype


#################   Start Timedelta Arithmetic Changes   ###################
# The following changes are needed to support datetime64 arr + timedelta64
# They are being merged into Numba in this PR:
# https://github.com/numba/numba/pull/7082


def mk_alloc(typemap, calltypes, lhs, size_var, dtype, scope, loc, lhs_typ):
    """generate an array allocation with np.empty() and return list of nodes.
    size_var can be an int variable or tuple of int variables.
    """
    # Imports to match Numba
    import numpy
    from numba.core import typing
    from numba.core.ir_utils import (
        convert_size_to_var,
        get_np_ufunc_typ,
        mk_unique_var,
    )

    out = []
    ndims = 1
    size_typ = types.intp
    if isinstance(size_var, tuple):
        if len(size_var) == 1:
            size_var = size_var[0]
            size_var = convert_size_to_var(size_var, typemap, scope, loc, out)
        else:
            # tuple_var = build_tuple([size_var...])
            ndims = len(size_var)
            tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
            if typemap:
                typemap[tuple_var.name] = types.containers.UniTuple(types.intp, ndims)
            # constant sizes need to be assigned to vars
            new_sizes = [
                convert_size_to_var(s, typemap, scope, loc, out) for s in size_var
            ]
            tuple_call = ir.Expr.build_tuple(new_sizes, loc)
            tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
            out.append(tuple_assign)
            size_var = tuple_var
            size_typ = types.containers.UniTuple(types.intp, ndims)
    # g_np_var = Global(numpy)
    g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
    if typemap:
        typemap[g_np_var.name] = types.misc.Module(numpy)
    g_np = ir.Global("np", numpy, loc)
    g_np_assign = ir.Assign(g_np, g_np_var, loc)
    # attr call: empty_attr = getattr(g_np_var, empty)
    empty_attr_call = ir.Expr.getattr(g_np_var, "empty", loc)
    attr_var = ir.Var(scope, mk_unique_var("$empty_attr_attr"), loc)
    if typemap:
        typemap[attr_var.name] = get_np_ufunc_typ(numpy.empty)
    attr_assign = ir.Assign(empty_attr_call, attr_var, loc)
    # Assume str(dtype) returns a valid type
    dtype_str = str(dtype)
    # alloc call: lhs = empty_attr(size_var, typ_var)
    typ_var = ir.Var(scope, mk_unique_var("$np_typ_var"), loc)
    if typemap:
        typemap[typ_var.name] = types.functions.NumberClass(dtype)
    # BODO CHANGE
    # If dtype is a datetime/timedelta with a unit,
    # then it won't return a valid type and instead can be created
    # with a string. i.e. "datetime64[ns]")
    if isinstance(dtype, (types.NPDatetime, types.NPTimedelta)) and dtype.unit != "":
        typename_const = ir.Const(dtype_str, loc)
        typ_var_assign = ir.Assign(typename_const, typ_var, loc)
    else:
        if dtype_str == "bool":
            # empty doesn't like 'bool' sometimes (e.g. kmeans example)
            dtype_str = "bool_"
        np_typ_getattr = ir.Expr.getattr(g_np_var, dtype_str, loc)
        typ_var_assign = ir.Assign(np_typ_getattr, typ_var, loc)
    alloc_call = ir.Expr.call(attr_var, [size_var, typ_var], (), loc)
    if calltypes:
        calltypes[alloc_call] = typemap[attr_var.name].get_call_type(
            typing.Context(), [size_typ, types.functions.NumberClass(dtype)], {}
        )
    # signature(
    #    types.npytypes.Array(dtype, ndims, 'C'), size_typ,
    #    types.functions.NumberClass(dtype))

    if lhs_typ.layout == "F":
        empty_c_typ = lhs_typ.copy(layout="C")
        empty_c_var = ir.Var(scope, mk_unique_var("$empty_c_var"), loc)
        if typemap:
            typemap[empty_c_var.name] = lhs_typ.copy(layout="C")
        empty_c_assign = ir.Assign(alloc_call, empty_c_var, loc)

        # attr call: asfortranarray = getattr(g_np_var, asfortranarray)
        asfortranarray_attr_call = ir.Expr.getattr(g_np_var, "asfortranarray", loc)
        afa_attr_var = ir.Var(scope, mk_unique_var("$asfortran_array_attr"), loc)
        if typemap:
            typemap[afa_attr_var.name] = get_np_ufunc_typ(numpy.asfortranarray)
        afa_attr_assign = ir.Assign(asfortranarray_attr_call, afa_attr_var, loc)
        # call asfortranarray
        asfortranarray_call = ir.Expr.call(afa_attr_var, [empty_c_var], (), loc)
        if calltypes:
            calltypes[asfortranarray_call] = typemap[afa_attr_var.name].get_call_type(
                typing.Context(), [empty_c_typ], {}
            )

        asfortranarray_assign = ir.Assign(asfortranarray_call, lhs, loc)

        out.extend(
            [
                g_np_assign,
                attr_assign,
                typ_var_assign,
                empty_c_assign,
                afa_attr_assign,
                asfortranarray_assign,
            ]
        )
    else:
        alloc_assign = ir.Assign(alloc_call, lhs, loc)
        out.extend([g_np_assign, attr_assign, typ_var_assign, alloc_assign])

    return out


if _check_numba_change:
    lines = inspect.getsource(numba.core.ir_utils.mk_alloc)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "c7d3fcf16b7f268082614da52b3a9ba8cb3a9e41b63595f915aa6e7ed13cbe63"
    ):  # pragma: no cover
        print(hashlib.sha256(lines.encode()).hexdigest())
        warnings.warn("mk_alloc has changed")

numba.core.ir_utils.mk_alloc = mk_alloc
# This function is imported in additional places
numba.parfors.parfor.mk_alloc = mk_alloc


def ufunc_find_matching_loop(ufunc, arg_types):
    """Find the appropriate loop to be used for a ufunc based on the types
    of the operands

    ufunc        - The ufunc we want to check
    arg_types    - The tuple of arguments to the ufunc, including any
                   explicit output(s).
    return value - A UFuncLoopSpec identifying the loop, or None
                   if no matching loop is found.
    """
    # Imports to match Numba
    import numpy as np
    from numba.np import npdatetime_helpers
    from numba.np.numpy_support import (
        UFuncLoopSpec,
        as_dtype,
        from_dtype,
        ufunc_can_cast,
    )

    # Separate logical input from explicit output arguments
    input_types = arg_types[: ufunc.nin]
    output_types = arg_types[ufunc.nin :]
    assert len(input_types) == ufunc.nin

    try:
        np_input_types = [as_dtype(x) for x in input_types]
    except NotImplementedError:
        return None
    try:
        np_output_types = [as_dtype(x) for x in output_types]
    except NotImplementedError:
        return None

    # Whether the inputs are mixed integer / floating-point
    has_mixed_inputs = any(dt.kind in "iu" for dt in np_input_types) and any(
        dt.kind in "cf" for dt in np_input_types
    )

    def choose_types(numba_types, ufunc_letters):
        """
        Return a list of Numba types representing *ufunc_letters*,
        except when the letter designates a datetime64 or timedelta64,
        in which case the type is taken from *numba_types*.
        """
        assert len(ufunc_letters) >= len(numba_types)
        types = [
            tp if letter in "mM" else from_dtype(np.dtype(letter))
            for tp, letter in zip(numba_types, ufunc_letters)
        ]
        # Add missing types (presumably implicit outputs)
        types += [
            from_dtype(np.dtype(letter)) for letter in ufunc_letters[len(numba_types) :]
        ]
        return types

    def set_output_dt_units(inputs, outputs, ufunc_inputs):
        """
        Sets the output unit of a datetime type based on the input units

        Timedelta is a special dtype that requires the time unit to be
        specified (day, month, etc). Not every operation with timedelta inputs
        leads to an output of timedelta output. However, for those that do,
        the unit of output must be inferred based on the units of the inputs.

        At the moment this function takes care of two cases:
        a) where all inputs are timedelta with the same unit (mm), and
        therefore the output has the same unit.
        This case is used for arr.sum, and for arr1+arr2 where all arrays
        are timedeltas.
        If in the future this needs to be extended to a case with mixed units,
        the rules should be implemented in `npdatetime_helpers` and called
        from this function to set the correct output unit.
        b) where left operand is a timedelta, i.e. the "m?" case. This case
        is used for division, eg timedelta / int.

        At the time of writing, Numba does not support addition of timedelta
        and other types, so this function does not consider the case "?m",
        i.e. where timedelta is the right operand to a non-timedelta left
        operand. To extend it in the future, just add another elif clause.
        """

        def make_specific(outputs, unit):
            new_outputs = []
            for out in outputs:
                if isinstance(out, types.NPTimedelta) and out.unit == "":
                    new_outputs.append(types.NPTimedelta(unit))
                else:
                    new_outputs.append(out)
            return new_outputs

        # BODO CHANGE 1:
        def make_datetime_specific(outputs, dt_unit, td_unit):
            new_outputs = []
            for out in outputs:
                if isinstance(out, types.NPDatetime) and out.unit == "":
                    unit = npdatetime_helpers.combine_datetime_timedelta_units(
                        dt_unit, td_unit
                    )
                    new_outputs.append(types.NPDatetime(unit))
                else:
                    new_outputs.append(out)
            return new_outputs

        if ufunc_inputs == "mm":
            if all(inp.unit == inputs[0].unit for inp in inputs):
                # Case with operation on same units. Operations on different
                # units not adjusted for now but might need to be
                # added in the future
                unit = inputs[0].unit
                new_outputs = make_specific(outputs, unit)
            else:
                return outputs
            return new_outputs
        elif ufunc_inputs == "mM":
            # case where the left operand has timedelta type
            # and the right operand has datetime
            td_unit = inputs[0].unit
            dt_unit = inputs[1].unit
            return make_datetime_specific(outputs, dt_unit, td_unit)

        elif ufunc_inputs == "Mm":
            # case where the right operand has timedelta type
            # and the left operand has datetime
            dt_unit = inputs[0].unit
            td_unit = inputs[1].unit
            return make_datetime_specific(outputs, dt_unit, td_unit)

        elif ufunc_inputs[0] == "m":
            # case where the left operand has timedelta type
            unit = inputs[0].unit
            new_outputs = make_specific(outputs, unit)
            return new_outputs

    # In NumPy, the loops are evaluated from first to last. The first one
    # that is viable is the one used. One loop is viable if it is possible
    # to cast every input operand to the one expected by the ufunc.
    # Also under NumPy 1.10+ the output must be able to be cast back
    # to a close enough type ("same_kind").

    for candidate in ufunc.types:
        ufunc_inputs = candidate[: ufunc.nin]
        ufunc_outputs = candidate[-ufunc.nout :] if ufunc.nout else []
        if "O" in ufunc_inputs:
            # Skip object arrays
            continue
        found = True
        # Skip if any input or output argument is mismatching
        for outer, inner in zip(np_input_types, ufunc_inputs):
            # (outer is a dtype instance, inner is a type char)
            if outer.char in "mM" or inner in "mM":
                # For datetime64 and timedelta64, we want to retain
                # precise typing (i.e. the units); therefore we look for
                # an exact match.
                if outer.char != inner:
                    found = False
                    break
            elif not ufunc_can_cast(outer.char, inner, has_mixed_inputs, "safe"):
                found = False
                break
        if found:
            # Can we cast the inner result to the outer result type?
            for outer, inner in zip(np_output_types, ufunc_outputs):
                if outer.char not in "mM" and not ufunc_can_cast(
                    inner, outer.char, has_mixed_inputs, "same_kind"
                ):
                    found = False
                    break
        if found:
            # Found: determine the Numba types for the loop's inputs and
            # outputs.
            try:
                inputs = choose_types(input_types, ufunc_inputs)
                outputs = choose_types(output_types, ufunc_outputs)
                # BODO CHANGE 2:
                # if the left operand or both are timedeltas, or we have
                # 1 datetime and 1 timedelta, then the output
                # units need to be determined.
                if ufunc_inputs[0] == "m" or ufunc_inputs == "Mm":
                    outputs = set_output_dt_units(inputs, outputs, ufunc_inputs)

            except NotImplementedError:
                # One of the selected dtypes isn't supported by Numba
                # (e.g. float16), try other candidates
                continue
            else:
                return UFuncLoopSpec(inputs, outputs, candidate)

    return None


if _check_numba_change:
    lines = inspect.getsource(numba.np.numpy_support.ufunc_find_matching_loop)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "31d1c4f9c2fb0dd0642bc3717e64f79a846e0dc5fdeecd36392cf546e4d7d85b"
    ):  # pragma: no cover
        print(hashlib.sha256(lines.encode()).hexdigest())
        warnings.warn("ufunc_find_matching_loop has changed")

numba.np.numpy_support.ufunc_find_matching_loop = ufunc_find_matching_loop
# This function is imported in additional places
numba.core.typing.npydecl.ufunc_find_matching_loop = ufunc_find_matching_loop
numba.np.ufunc.gufunc.ufunc_find_matching_loop = ufunc_find_matching_loop
import numba.np.npyimpl

numba.np.npyimpl.ufunc_find_matching_loop = ufunc_find_matching_loop

#################   End Timedelta Arithmetic Changes   ###################
