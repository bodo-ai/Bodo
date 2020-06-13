"""
Numba monkey patches to fix issues related to Bodo. Should be imported before any
other module in bodo package.
"""
import operator
import functools
import hashlib
import sys
import os
import re
import inspect
import warnings
import numpy as np

import numba
from numba.core import ir, ir_utils, types, analysis, errors
from numba.core.utils import reraise
from numba.core.typing.typeof import Purpose, typeof
from numba.core.ir_utils import (
    compile_to_numba_ir,
    replace_arg_nodes,
    find_const,
    guard,
    GuardException,
    get_definition,
    require,
    find_callname,
    build_definitions,
    remove_dead_extensions,
    has_no_side_effect,
    analysis,
    visit_vars_extensions,
    visit_vars_inner,
)
from numba.extending import lower_builtin
import numba.np.linalg
from numba.core.imputils import impl_ret_new_ref
from numba.core.typing.templates import infer_global, signature
from numba.core.types.misc import unliteral
from numba.core.typing.templates import (
    AbstractTemplate,
    _OverloadAttributeTemplate,
    _OverloadMethodTemplate,
)
from numba.core.types.functions import _ResolutionFailures
from numba.core.errors import LiteralTypingError
from numba.core.types import literal


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


# make sure visit_vars_stmt hasn't changed before replacing it
lines = inspect.getsource(numba.core.ir_utils.visit_vars_stmt)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "52b7b645ba65c35f3cf564f936e113261db16a2dff1e80fbee2459af58844117"
):  # pragma: no cover
    warnings.warn("numba.core.ir_utils.visit_vars_stmt has changed")


numba.core.ir_utils.visit_vars_stmt = visit_vars_stmt


# The code below is copied from Numba and modified to handle aliases with tuple values.
# https://github.com/numba/numba/blob/cc7e7c7cfa6389b54d3b5c2c95751c97eb531a96/numba/ir_utils.py#L725
# This case happens for Bodo dataframes since init_dataframe takes a tuple of arrays as
# input, and output dataframe is aliased with all of these arrays. see test_df_alias.
from numba.core.ir_utils import (
    _add_alias,
    alias_analysis_extensions,
    alias_func_extensions,
)
import copy


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
                    and expr.value.name in arg_aliases
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


# make sure find_potential_aliases hasn't changed before replacing it
lines = inspect.getsource(ir_utils.find_potential_aliases)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "ea2b49b83066d0dca57c8e62202fa6b438b429668b010fc1e9580e7bedfb1a70"
):  # pragma: no cover
    warnings.warn("ir_utils.find_potential_aliases has changed")


ir_utils.find_potential_aliases = find_potential_aliases


# The code below is copied from Numba and modified to fix Numba #5539.
# TODO: remove when the issue is fixed
# https://github.com/numba/numba/blob/afd5c67b1ed6f51c040d1845a014abea8b87846a/numba/np/linalg.py#L462
def dot_2_vm(context, builder, sig, args):
    """
    np.dot(vector, matrix)
    """

    def dot_impl(a, b):  # pragma: no cover
        (m,) = a.shape
        _m, n = b.shape
        # changed code: initialize with zeros if inputs are empty
        if m == 0:
            out = np.zeros((n,), a.dtype)
        else:
            out = np.empty((n,), a.dtype)
        return np.dot(a, b, out)

    res = context.compile_internal(builder, dot_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)


# make sure numba.np.linalg.dot_2_vm hasn't changed before replacing it
lines = inspect.getsource(numba.np.linalg.dot_2_vm)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "85c6fa1687773b818f76b08b2e43cf6251d5cc87ac64906434ba5dfaa0bcc3cb"
):  # pragma: no cover
    warnings.warn("numba.np.linalg.dot_2_vm has changed")


numba.np.linalg.dot_2_vm = dot_2_vm


# replace Numba's overload/overload_method handling functions to support a new option
# called 'no_unliteral', which avoids a second run of overload with literal types
# converted to non-literal versions. This solves hiding errors such as #889
# TODO: remove after Numba's #5411 is resolved
_overload_default_jit_options = {"no_cpython_wrapper": True}


# change: added no_unliteral argument
def overload(func, jit_options={}, strict=True, inline="never", no_unliteral=False):
    from numba.core.typing.templates import make_overload_template, infer_global, infer

    # set default options
    opts = _overload_default_jit_options.copy()
    opts.update(jit_options)  # let user options override

    def decorate(overload_func):
        # change: added no_unliteral argument
        template = make_overload_template(
            func, overload_func, opts, strict, inline, no_unliteral
        )
        infer(template)
        if callable(func):
            infer_global(func, types.Function(template))
        return overload_func

    return decorate


# make sure overload hasn't changed before replacing it
lines = inspect.getsource(numba.core.extending.overload)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "dbcb7029c1538d6ee6324ae6a6d787527cf840997a4464d3053f46af0cd696b2"
):  # pragma: no cover
    warnings.warn("numba.core.extending.overload has changed")


numba.core.extending.overload = overload
numba.extending.overload = overload


def overload_method(typ, attr, **kwargs):
    from numba.core.typing.templates import make_overload_method_template, infer_getattr

    def decorate(overload_func):
        template = make_overload_method_template(
            typ,
            attr,
            overload_func,
            inline=kwargs.get("inline", "never"),
            # change: added no_unliteral argument
            no_unliteral=kwargs.get("no_unliteral", False),
        )
        infer_getattr(template)
        overload(overload_func, **kwargs)(overload_func)
        return overload_func

    return decorate


# make sure overload_method hasn't changed before replacing it
lines = inspect.getsource(numba.core.extending.overload_method)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "628441a93db4b93feae28e78a027a7682933a95936ff3afdc2d64b98388d7d95"
):  # pragma: no cover
    warnings.warn("numba.core.extending.overload_method has changed")


numba.core.extending.overload_method = overload_method
numba.extending.overload_method = overload_method


from numba.core.cpu_options import InlineOptions


# change: added no_unliteral argument
def make_overload_template(
    func, overload_func, jit_options, strict, inline, no_unliteral
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
        # change: added no_unliteral argument
        _inline_overloads={},
        _no_unliteral=no_unliteral,
    )
    return type(base)(name, (base,), dct)


# make sure make_overload_template hasn't changed before replacing it
lines = inspect.getsource(numba.core.typing.templates.make_overload_template)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "b62d5f58dbfeb9753e8c6c94bdf6ffb9ffa39eb5c34bd2c2bea2be2a89c8d7ec"
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
        # change: added _no_unliteral attribute
        _no_unliteral = getattr(self, "_no_unliteral", False)
        _overload_func = staticmethod(self._overload_func)
        _inline_overloads = self._inline_overloads

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


# make sure _resolve hasn't changed before replacing it
lines = inspect.getsource(numba.core.typing.templates._OverloadMethodTemplate._resolve)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "40645505764f3f6fc52c8b479cbb4dc6203025fa409d95177f9dad89569c118e"
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
    no_unliteral=False,
    base=_OverloadAttributeTemplate,
):
    """
    Make a template class for attribute *attr* of *typ* overloaded by
    *overload_func*.
    """
    assert isinstance(typ, types.Type) or issubclass(typ, types.Type)
    name = "OverloadTemplate_%s_%s" % (typ, attr)
    # Note the implementation cache is subclass-specific
    dct = dict(
        key=typ,
        _attr=attr,
        _impl_cache={},
        _inline=staticmethod(InlineOptions(inline)),
        _inline_overloads={},
        # change: added _no_unliteral argument
        _no_unliteral=no_unliteral,
        _overload_func=staticmethod(overload_func),
    )
    return type(base)(name, (base,), dct)


# make sure make_overload_attribute_template hasn't changed before replacing it
lines = inspect.getsource(numba.core.typing.templates.make_overload_attribute_template)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "e85ed81e6a6bceb09ee2cee43b5a2d4c11a2805e29e2f60f37fe49b2b9996f55"
):  # pragma: no cover
    warnings.warn(
        "numba.core.typing.templates.make_overload_attribute_template has changed"
    )


numba.core.typing.templates.make_overload_attribute_template = (
    make_overload_attribute_template
)


# change: added no_unliteral argument
def make_overload_method_template(typ, attr, overload_func, inline, no_unliteral):
    """
    Make a template class for method *attr* of *typ* overloaded by
    *overload_func*.
    """
    return make_overload_attribute_template(
        # change: added no_unliteral argument
        typ,
        attr,
        overload_func,
        inline=inline,
        no_unliteral=no_unliteral,
        base=_OverloadMethodTemplate,
    )


# make sure make_overload_method_template hasn't changed before replacing it
lines = inspect.getsource(numba.core.typing.templates.make_overload_method_template)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "899327427f7db757cfdd6d5d916f81912831e8e41f8dc07436790254502ffc27"
):  # pragma: no cover
    warnings.warn(
        "numba.core.typing.templates.make_overload_method_template has changed"
    )


numba.core.typing.templates.make_overload_method_template = (
    make_overload_method_template
)


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


# make sure bound_function hasn't changed before replacing it
lines = inspect.getsource(numba.core.typing.templates.bound_function)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "a2feefe64eae6a15c56affc47bf0c1d04461f9566913442d539452b397103322"
):  # pragma: no cover
    warnings.warn("numba.core.typing.templates.bound_function has changed")


numba.core.typing.templates.bound_function = bound_function


def get_call_type(self, context, args, kws):
    failures = _ResolutionFailures(context, self, args, kws)
    for temp_cls in self.templates:
        temp = temp_cls(context)
        for uselit in [True, False]:
            try:
                if uselit:
                    sig = temp.apply(args, kws)
                # change: check _no_unliteral attribute if present
                elif not getattr(temp, "_no_unliteral", False):
                    nolitargs = tuple([unliteral(a) for a in args])
                    nolitkws = {k: unliteral(v) for k, v in kws.items()}
                    sig = temp.apply(nolitargs, nolitkws)
            except Exception as e:
                sig = None
                failures.add_error(temp_cls, e)
            else:
                if sig is not None:
                    self._impl_keys[sig.args] = temp.get_impl_key(sig)
                    return sig
                else:
                    haslit = "" if uselit else "out"
                    msg = "All templates rejected with%s literals." % haslit
                    failures.add_error(temp_cls, msg)

    if len(failures) == 0:
        raise AssertionError(
            "Internal Error. "
            "Function resolution ended with no failures "
            "or successful signature"
        )
    failures.raise_error()


# make sure get_call_type hasn't changed before replacing it
lines = inspect.getsource(numba.core.types.functions.BaseFunction.get_call_type)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "bcb57ef2f0557836bf15c69eb09ffb16955633eb86781c73bcde0e4910fb0d06"
):  # pragma: no cover
    warnings.warn("numba.core.types.functions.BaseFunction.get_call_type has changed")


numba.core.types.functions.BaseFunction.get_call_type = get_call_type


def get_call_type2(self, context, args, kws):
    template = self.template(context)
    e = None
    # Try with Literal
    try:
        out = template.apply(args, kws)
    except Exception:
        out = None
    # If that doesn't work, remove literals
    # change: check _no_unliteral attribute if present
    if out is None and not getattr(template, "_no_unliteral", False):
        args = [unliteral(a) for a in args]
        kws = {k: unliteral(v) for k, v in kws.items()}
        out = template.apply(args, kws)
    if out is None and e is not None:
        raise e
    return out


# make sure get_call_type hasn't changed before replacing it
lines = inspect.getsource(numba.core.types.functions.BoundFunction.get_call_type)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "9c665cf809ee7310608ce667e0173a53fbfc2e804b85ba02a98b88062c9e77e0"
):  # pragma: no cover
    warnings.warn("numba.core.types.functions.BoundFunction.get_call_type has changed")


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
# method to delete args before returning the dispatcher object. Otherwise
# the code is the same
def _compile_for_args(self, *args, **kws):  # pragma: no cover
    """
    For internal use.  Compile a specialized version of the function
    for the given *args* and *kws*, and return the resulting callable.
    """
    assert not kws
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
            reraise(type(e), e, None)

    argtypes = []
    for a in args:
        if isinstance(a, numba.core.dispatcher.OmittedArg):
            argtypes.append(types.Omitted(a.value))
        else:
            argtypes.append(self.typeof_pyval(a))
    try:
        error = None
        return self.compile(tuple(argtypes))
    except errors.ForceLiteralArg as e:
        # Received request for compiler re-entry with the list of arguments
        # indicated by e.requested_args.
        # First, check if any of these args are already Literal-ized
        already_lit_pos = [
            i for i in e.requested_args if isinstance(args[i], types.Literal)
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
        args = [
            (types.literal if i in e.requested_args else lambda x: x)(args[i])
            for i, v in enumerate(args)
        ]
        # Re-enter compilation with the Literal-ized arguments
        return self._compile_for_args(*args)

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
    except Exception as e:
        if numba.core.config.SHOW_HELP:
            if hasattr(e, "patch_message"):
                help_msg = errors.error_extras["reportable"]
                e.patch_message("\n".join((str(e).rstrip(), help_msg)))
        # ignore the FULL_TRACEBACKS config, this needs reporting!
        raise e
    except bodo.utils.typing.BodoError as e:
        # create a new error so that the stacktrace only reaches
        # the point where the new error is raised
        error = bodo.utils.typing.BodoError(str(e))
    finally:
        # avoid issue of reference leak of arguments to jitted function:
        # https://github.com/numba/numba/issues/5419
        del args
        if error:
            raise error


# workaround for Numba #5419 issue (https://github.com/numba/numba/issues/5419)
# first we check that the hash of the Numba function that we are replacing
# matches the one of the function that we copied from Numba

lines = inspect.getsource(numba.core.dispatcher._DispatcherBase._compile_for_args)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "1e12bb18f3ed09e608ba6c56a7fcd4cf2fe342b71af9d2e9767aee817d92f4b8"
):  # pragma: no cover
    warnings.warn("numba.core.dispatcher._DispatcherBase._compile_for_args has changed")
# now replace the function with our own
numba.core.dispatcher._DispatcherBase._compile_for_args = _compile_for_args


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
                    str(e), loc=constraint.loc, highlighting=False,
                )
                errors.append(numba.core.utils.chain_exception(new_exc, e))
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
            except bodo.utils.typing.BodoError as e:
                if e.is_new:
                    # the first time we see BodoError during type inference, we
                    # put the code location in the error message, and re-raise
                    loc = constraint.loc
                    raise bodo.utils.typing.BodoError(
                        str(e) + "\n" + loc.strformat() + "\n", is_new=False
                    )
                else:
                    # keep raising and propagating the error through numba until
                    # it reaches the user
                    raise e
    return errors


lines = inspect.getsource(numba.core.typeinfer.ConstraintNetwork.propagate)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "2c204df4d8c58da7c86e0abbab48a7a7863ee3cbe8d2ba89f617d4f580b622e9"
):  # pragma: no cover
    warnings.warn("numba.core.typeinfer.ConstraintNetwork.propagate has changed")
numba.core.typeinfer.ConstraintNetwork.propagate = propagate


# replaces remove_dead_block of Numba to add Bodo optimization (e.g. replace dead array
# in array.shape)
def bodo_remove_dead_block(
    block, lives, call_table, arg_aliases, alias_map, alias_set, func_ir, typemap
):
    """remove dead code using liveness info.
    Mutable arguments (e.g. arrays) that are not definitely assigned are live
    after return of function.
    """
    import bodo
    from bodo.utils.utils import is_array_typ, is_expr
    from bodo.transforms.distributed_pass import saved_array_analysis

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
        self.typingctx, self.func_ir, self.typemap, self.calltypes,
    )

    # bodo change: make sure _max_label is always maximum
    ir_utils._max_label = max(ir_utils._max_label, max(func_ir.blocks.keys()))
    self.flags = flags


lines = inspect.getsource(numba.parfors.parfor.ParforPassStates.__init__)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "0b939cec777ef428a224056d8ab92db860a2b1457fd228127b695d02df933c26"
):  # pragma: no cover
    warnings.warn("numba.parfors.parfor.ParforPassStates.__init__ has changed")

numba.parfors.parfor.ParforPassStates.__init__ = ParforPassStates__init__


# replace Numba's maybe_literal to avoid using our ListLiteral in type inference
def maybe_literal(value):
    """Get a Literal type for the value or None.
    """
    # bodo change: don't use our ListLiteral for regular constant or global lists.
    # ListLiteral is only used when Bodo forces an argument to be a literal
    if isinstance(value, list):
        return
    try:
        return literal(value)
    except LiteralTypingError:
        return


lines = inspect.getsource(types.maybe_literal)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "8fb2fd93acf214b28e33e37d19dc2f7290a42792ec59b650553ac278854b5081"
):  # pragma: no cover
    warnings.warn("types.maybe_literal has changed")

types.maybe_literal = maybe_literal
types.misc.maybe_literal = maybe_literal


def CacheImpl__init__(self, py_func):
    self._is_closure = bool(py_func.__closure__)
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


lines = inspect.getsource(numba.core.caching._CacheImpl.__init__)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "f84d6f319647b4eb905b1b59b576772a4caf7655a5bd094405f4f40ccb7a9c95"
):  # pragma: no cover
    warnings.warn("numba.core.caching._CacheImpl.__init__ has changed")

numba.core.caching._CacheImpl.__init__ = CacheImpl__init__


# replacing _analyze_broadcast in array analysis to fix a bug. It's assuming that
# get_shape throws GuardException which is wrong.
# Numba 0.48 exposed this error with test_linear_regression since array analysis is
# more restrictive and assumes more variables as redefined
def _analyze_broadcast(self, scope, equiv_set, loc, args):
    """Infer shape equivalence of arguments based on Numpy broadcast rules
    and return shape of output
    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    """
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
    #     return arrs[0], self._call_assert_equiv(scope, loc, equiv_set, arrs)
    shapes = [equiv_set.get_shape(x) for x in arrs]
    if any(a is None for a in shapes):
        return arrs[0], self._call_assert_equiv(scope, loc, equiv_set, arrs)
    return self._broadcast_assert_shapes(scope, equiv_set, loc, shapes, names)


lines = inspect.getsource(numba.parfors.array_analysis.ArrayAnalysis._analyze_broadcast)
if (
    hashlib.sha256(lines.encode()).hexdigest()
    != "7dd54560e6af49661182672532f711e3eb643b99a12ed847b0ed580ffe60f702"
):  # pragma: no cover
    warnings.warn(
        "numba.parfors.array_analysis.ArrayAnalysis._analyze_broadcast has changed"
    )

numba.parfors.array_analysis.ArrayAnalysis._analyze_broadcast = _analyze_broadcast
