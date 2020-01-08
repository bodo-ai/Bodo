# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Collection of utility functions. Needs to be refactored in separate files.
"""
from collections import namedtuple
import operator
import keyword
import numba
from numba import ir_utils, ir, types, cgutils
from numba.ir_utils import (
    guard,
    get_definition,
    find_callname,
    require,
    add_offset_to_labels,
    find_topo_order,
    find_const,
    mk_unique_var,
)
from numba.parfor import wrap_parfor_blocks, unwrap_parfor_blocks
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.targets.imputils import lower_builtin
from numba.extending import overload, intrinsic, lower_cast
import collections
import numpy as np
import bodo
from bodo.libs.str_ext import string_type
from bodo.libs.list_str_arr_ext import list_string_array_type
from bodo.libs.str_arr_ext import (
    string_array_type,
    num_total_chars,
    pre_alloc_string_array,
    get_utf8_size,
)
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import boolean_array
from enum import Enum


# int values for types to pass to C code
# XXX: _bodo_common.h
class CTypeEnum(Enum):
    Int8 = 0
    UInt8 = 1
    Int32 = 2
    UInt32 = 3
    Int64 = 4
    UInt64 = 7
    Float32 = 5
    Float64 = 6
    Int16 = 8
    UInt16 = 9


_numba_to_c_type_map = {
    types.int8: CTypeEnum.Int8.value,
    types.uint8: CTypeEnum.UInt8.value,
    types.int32: CTypeEnum.Int32.value,
    types.uint32: CTypeEnum.UInt32.value,
    types.int64: CTypeEnum.Int64.value,
    types.uint64: CTypeEnum.UInt64.value,
    types.float32: CTypeEnum.Float32.value,
    types.float64: CTypeEnum.Float64.value,
    types.NPDatetime("ns"): CTypeEnum.UInt64.value,
    types.NPTimedelta("ns"): CTypeEnum.UInt64.value,
    # XXX: Numpy's bool array uses a byte for each value but regular booleans
    # are not bytes
    # TODO: handle boolean scalars properly
    types.bool_: CTypeEnum.UInt8.value,
    types.int16: CTypeEnum.Int16.value,
    types.uint16: CTypeEnum.UInt16.value,
}


# silence Numba error messages for now
# TODO: customize through @bodo.jit
numba.errors.error_extras = {
    "unsupported_error": "",
    "typing": "",
    "reportable": "",
    "interpreter": "",
    "constant_inference": "",
}

# sentinel value representing non-constant values
class NotConstant:
    pass


NOT_CONSTANT = NotConstant()

ReplaceFunc = namedtuple(
    "ReplaceFunc", ["func", "arg_types", "args", "glbls", "pre_nodes"]
)

np_alloc_callnames = ("empty", "zeros", "ones", "full")


def unliteral_all(args):
    return tuple(types.unliteral(a) for a in args)


# TODO: move to Numba
class BooleanLiteral(types.Literal, types.Boolean):
    def can_convert_to(self, typingctx, other):
        # similar to IntegerLiteral
        conv = typingctx.can_convert(self.literal_type, other)
        if conv is not None:
            return max(conv, types.Conversion.promote)


types.Literal.ctor_map[bool] = BooleanLiteral

numba.datamodel.register_default(BooleanLiteral)(numba.extending.models.BooleanModel)


@lower_cast(BooleanLiteral, types.Boolean)
def literal_bool_cast(context, builder, fromty, toty, val):
    lit = context.get_constant_generic(
        builder, fromty.literal_type, fromty.literal_value
    )
    return context.cast(builder, lit, fromty.literal_type, toty)


def get_constant(func_ir, var, default=NOT_CONSTANT):
    def_node = guard(get_definition, func_ir, var)
    if def_node is None:
        return default
    if isinstance(def_node, ir.Const):
        return def_node.value
    # call recursively if variable assignment
    if isinstance(def_node, ir.Var):
        return get_constant(func_ir, def_node, default)
    return default


def inline_new_blocks(func_ir, block, i, callee_blocks, work_list=None):
    # adopted from inline_closure_call
    scope = block.scope
    instr = block.body[i]

    # 1. relabel callee_ir by adding an offset
    callee_blocks = add_offset_to_labels(callee_blocks, ir_utils._max_label + 1)
    callee_blocks = ir_utils.simplify_CFG(callee_blocks)
    max_label = max(callee_blocks.keys())
    #    reset globals in ir_utils before we use it
    ir_utils._max_label = max_label
    topo_order = find_topo_order(callee_blocks)

    # 5. split caller blocks into two
    new_blocks = []
    new_block = ir.Block(scope, block.loc)
    new_block.body = block.body[i + 1 :]
    new_label = ir_utils.next_label()
    func_ir.blocks[new_label] = new_block
    new_blocks.append((new_label, new_block))
    block.body = block.body[:i]
    min_label = topo_order[0]
    block.body.append(ir.Jump(min_label, instr.loc))

    # 6. replace Return with assignment to LHS
    numba.inline_closurecall._replace_returns(callee_blocks, instr.target, new_label)
    #    remove the old definition of instr.target too
    if instr.target.name in func_ir._definitions:
        func_ir._definitions[instr.target.name] = []

    # 7. insert all new blocks, and add back definitions
    for label in topo_order:
        # block scope must point to parent's
        block = callee_blocks[label]
        block.scope = scope
        numba.inline_closurecall._add_definitions(func_ir, block)
        func_ir.blocks[label] = block
        new_blocks.append((label, block))

    if work_list is not None:
        for block in new_blocks:
            work_list.append(block)
    return callee_blocks


def is_alloc_call(func_var, call_table):
    """
    return true if func_var represents an array creation call
    """
    assert func_var in call_table
    call_list = call_table[func_var]
    return (
        len(call_list) == 2
        and call_list[1] == np
        and call_list[0] in ["empty", "zeros", "ones", "full"]
    ) or call_list == [numba.unsafe.ndarray.empty_inferred]


def is_alloc_callname(func_name, mod_name):
    """
    return true if function represents an array creation call
    """
    return isinstance(mod_name, str) and (
        (mod_name == "numpy" and func_name in np_alloc_callnames)
        or (
            func_name == "empty_inferred"
            and mod_name in ("numba.extending", "numba.unsafe.ndarray")
        )
        or (
            func_name == "pre_alloc_string_array"
            and mod_name == "bodo.libs.str_arr_ext"
        )
        or (func_name == "alloc_str_list" and mod_name == "bodo.libs.str_ext")
        or (
            func_name == "pre_alloc_list_string_array"
            and mod_name == "bodo.libs.list_str_arr_ext"
        )
        or (func_name == "alloc_bool_array" and mod_name == "bodo.libs.bool_arr_ext")
        or (func_name == "alloc_datetime_date_array"
            and mod_name == "bodo.hiframes.datetime_date_ext")
    )


def find_build_tuple(func_ir, var):
    """Check if a variable is constructed via build_tuple
    and return the sequence or raise GuardException otherwise.
    """
    # variable or variable name
    require(isinstance(var, (ir.Var, str)))
    var_def = get_definition(func_ir, var)
    require(isinstance(var_def, ir.Expr))
    require(var_def.op == "build_tuple")
    return var_def.items


def cprint(*s):
    print(*s)


@infer_global(cprint)
class CprintInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.none, *unliteral_all(args))


typ_to_format = {
    types.int32: "d",
    types.uint32: "u",
    types.int64: "lld",
    types.uint64: "llu",
    types.float32: "f",
    types.float64: "lf",
}

from llvmlite import ir as lir
import llvmlite.binding as ll
from bodo.libs import hstr_ext

ll.add_symbol("print_str", hstr_ext.print_str)
ll.add_symbol("print_char", hstr_ext.print_char)


@lower_builtin(cprint, types.VarArg(types.Any))
def cprint_lower(context, builder, sig, args):
    from bodo.libs.str_ext import string_type, char_type

    for i, val in enumerate(args):
        typ = sig.args[i]
        if typ == string_type:
            fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
            fn = builder.module.get_or_insert_function(fnty, name="print_str")
            builder.call(fn, [val])
            cgutils.printf(builder, " ")
            continue
        if typ == char_type:
            fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8)])
            fn = builder.module.get_or_insert_function(fnty, name="print_char")
            builder.call(fn, [val])
            cgutils.printf(builder, " ")
            continue
        if isinstance(typ, types.ArrayCTypes):
            cgutils.printf(builder, "%p ", val)
            continue
        format_str = typ_to_format[typ]
        cgutils.printf(builder, "%{} ".format(format_str), val)
    cgutils.printf(builder, "\n")
    return context.get_dummy_value()


def print_dist(d):
    from bodo.transforms.distributed_analysis import Distribution

    if d == Distribution.REP:
        return "REP"
    if d == Distribution.OneD:
        return "1D_Block"
    if d == Distribution.OneD_Var:
        return "1D_Block_Var"
    if d == Distribution.Thread:
        return "Multi-thread"
    if d == Distribution.TwoD:
        return "2D_Block"


def distribution_report():
    import bodo.transforms.distributed_pass

    if bodo.transforms.distributed_pass.dist_analysis is None:
        return
    print("Array distributions:")
    for arr, dist in bodo.transforms.distributed_pass.dist_analysis.array_dists.items():
        print("   {0:20} {1}".format(arr, print_dist(dist)))
    print("\nParfor distributions:")
    for p, dist in bodo.transforms.distributed_pass.dist_analysis.parfor_dists.items():
        print("   {0:<20} {1}".format(p, print_dist(dist)))


def is_whole_slice(typemap, func_ir, var, accept_stride=False):
    """ return True if var can be determined to be a whole slice """
    require(
        typemap[var.name] == types.slice2_type
        or (accept_stride and typemap[var.name] == types.slice3_type)
    )
    call_expr = get_definition(func_ir, var)
    require(isinstance(call_expr, ir.Expr) and call_expr.op == "call")
    assert len(call_expr.args) == 2 or (accept_stride and len(call_expr.args) == 3)
    assert find_callname(func_ir, call_expr) == ("slice", "builtins")
    arg0_def = get_definition(func_ir, call_expr.args[0])
    arg1_def = get_definition(func_ir, call_expr.args[1])
    require(isinstance(arg0_def, ir.Const) and arg0_def.value == None)
    require(isinstance(arg1_def, ir.Const) and arg1_def.value == None)
    return True


def is_const_slice(typemap, func_ir, var, accept_stride=False):
    """ return True if var can be determined to be a constant size slice """
    require(
        typemap[var.name] == types.slice2_type
        or (accept_stride and typemap[var.name] == types.slice3_type)
    )
    call_expr = get_definition(func_ir, var)
    require(isinstance(call_expr, ir.Expr) and call_expr.op == "call")
    assert len(call_expr.args) == 2 or (accept_stride and len(call_expr.args) == 3)
    assert find_callname(func_ir, call_expr) == ("slice", "builtins")
    arg0_def = get_definition(func_ir, call_expr.args[0])
    require(isinstance(arg0_def, ir.Const) and arg0_def.value == None)
    size_const = find_const(func_ir, call_expr.args[1])
    require(isinstance(size_const, int))
    return True


def get_slice_step(typemap, func_ir, var):
    require(typemap[var.name] == types.slice3_type)
    call_expr = get_definition(func_ir, var)
    require(isinstance(call_expr, ir.Expr) and call_expr.op == "call")
    assert len(call_expr.args) == 3
    return call_expr.args[2]


def is_array_typ(var_typ):
    # TODO: make sure all Bodo arrays are here
    return (
        is_np_array_typ(var_typ)
        or var_typ
        in (
            string_array_type,
            list_string_array_type,
            bodo.hiframes.split_impl.string_array_split_view_type,
            bodo.hiframes.datetime_date_ext.datetime_date_array_type,
        )
        or isinstance(var_typ, bodo.hiframes.pd_series_ext.SeriesType)
        or bodo.hiframes.pd_index_ext.is_pd_index_type(var_typ)
        or isinstance(var_typ, IntegerArrayType)
        or var_typ == boolean_array
    )


def is_np_array_typ(var_typ):
    return isinstance(var_typ, types.Array)


def is_array_container_typ(var_typ):
    return isinstance(var_typ, (types.List, types.Set)) and is_array_typ(var_typ.dtype)


# TODO: fix tuple, dataframe distribution
def is_distributable_typ(var_typ):
    return (
        is_array_typ(var_typ)
        or isinstance(var_typ, bodo.hiframes.pd_dataframe_ext.DataFrameType)
        or (
            isinstance(var_typ, (types.List, types.Set))
            and is_distributable_typ(var_typ.dtype)
        )
    )


def is_distributable_tuple_typ(var_typ):
    return (
        isinstance(var_typ, types.BaseTuple)
        and any(
            is_distributable_typ(t) or is_distributable_tuple_typ(t)
            for t in var_typ.types
        )
    ) or (
        isinstance(var_typ, (types.List, types.Set))
        and is_distributable_tuple_typ(var_typ.dtype)
    )


@numba.generated_jit(nopython=True, cache=True)
def build_set(A):
    # if isinstance(A, IntegerArrayType):
    #     #return lambda A: set(A._data)
    #     def impl_int_arr(A):
    #         s = set()
    #         for i in range(len(A)):
    #             if not bodo.libs.array_kernels.isna(A, i):
    #                 s.add(A[i])
    #         return s

    #     return impl_int_arr
    # else:
    #     return lambda A: set(A)

    # TODO: use more efficient hash table optimized for addition and
    # membership check
    # XXX using dict for now due to Numba's #4577
    # avoid value if NA is not sentinel like np.nan
    if isinstance(A, IntegerArrayType) or A in (string_array_type, boolean_array):

        def impl_int_arr(A):  # pragma: no cover
            s = dict()
            for i in range(len(A)):
                if not bodo.libs.array_kernels.isna(A, i):
                    s[A[i]] = 0
            return s

        return impl_int_arr
    else:

        def impl(A):  # pragma: no cover
            s = dict()
            for i in range(len(A)):
                s[A[i]] = 0
            return s

        return impl


# converts an iterable to array, similar to np.array, but can support
# other things like StringArray
# TODO: other types like datetime?
def to_array(A):  # pragma: no cover
    return np.array(A)


@overload(to_array)
def to_array_overload(A):
    # handle dict for set replacement workaround
    if isinstance(A, types.DictType):
        dtype = A.key_type
        if dtype == string_type:

            def impl_str(A):  # pragma: no cover
                n = len(A)
                n_char = 0
                for v in A.keys():
                    n_char += get_utf8_size(v)
                arr = pre_alloc_string_array(n, n_char)
                i = 0
                for v in A.keys():
                    arr[i] = v
                    i += 1
                return arr

            return impl_str

        def impl(A):  # pragma: no cover
            n = len(A)
            arr = np.empty(n, dtype)
            i = 0
            for v in A.keys():
                arr[i] = v
                i += 1
            return arr

        return impl
    # try regular np.array and return it if it works
    def to_array_impl(A):  # pragma: no cover
        return np.array(A)

    try:
        numba.njit(to_array_impl).get_call_template((A,), {})
        return to_array_impl
    except:
        pass  # should be handled elsewhere (e.g. Set)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def unique(A):
    if isinstance(A, IntegerArrayType) or A == boolean_array:
        return lambda A: A.unique()

    # TODO: preserve order
    return lambda A: to_array(build_set(A))


@overload(np.array)
def np_array_array_overload(A):
    if isinstance(A, types.Array):
        return lambda A: A

    if isinstance(A, types.containers.Set):
        # TODO: naive implementation, data from set can probably
        # be copied to array more efficienty
        dtype = A.dtype

        def f(A):  # pragma: no cover
            n = len(A)
            arr = np.empty(n, dtype)
            i = 0
            for a in A:
                arr[i] = a
                i += 1
            return arr

        return f


def empty_like_type(n, arr):  # pragma: no cover
    return np.empty(n, arr.dtype)


@overload(empty_like_type)
def empty_like_type_overload(n, arr):
    # categorical
    if isinstance(arr, bodo.hiframes.pd_categorical_ext.CategoricalArray):
        from bodo.hiframes.pd_categorical_ext import fix_cat_array_type

        return lambda n, arr: fix_cat_array_type(np.empty(n, arr.dtype))

    if isinstance(arr, types.Array):
        return lambda n, arr: np.empty(n, arr.dtype)

    if isinstance(arr, types.List) and arr.dtype == string_type:

        def empty_like_type_str_list(n, arr):
            return [""] * n

        return empty_like_type_str_list

    # nullable int arr
    if isinstance(arr, IntegerArrayType):
        _dtype = arr.dtype

        def empty_like_type_int_arr(n, arr):
            n_bytes = (n + 7) >> 3
            return bodo.libs.int_arr_ext.init_integer_array(
                np.empty(n, _dtype), np.empty(n_bytes, np.uint8)
            )

        return empty_like_type_int_arr

    if arr == boolean_array:

        def empty_like_type_bool_arr(n, arr):
            n_bytes = (n + 7) >> 3
            return bodo.libs.bool_arr_ext.init_bool_array(
                np.empty(n, np.bool_), np.empty(n_bytes, np.uint8)
            )

        return empty_like_type_bool_arr

    # string array buffer for join
    assert arr == string_array_type

    def empty_like_type_str_arr(n, arr):  # pragma: no cover
        # average character heuristic
        avg_chars = 20  # heuristic
        if len(arr) != 0:
            avg_chars = num_total_chars(arr) // len(arr)
        return pre_alloc_string_array(n, n * avg_chars)

    return empty_like_type_str_arr


def alloc_arr_tup(n, arr_tup, init_vals=()):  # pragma: no cover
    arrs = []
    for in_arr in arr_tup:
        arrs.append(np.empty(n, in_arr.dtype))
    return tuple(arrs)


@overload(alloc_arr_tup)
def alloc_arr_tup_overload(n, data, init_vals=()):
    count = data.count

    allocs = ",".join(["empty_like_type(n, data[{}])".format(i) for i in range(count)])

    if init_vals is not ():
        # TODO check for numeric value
        allocs = ",".join(
            [
                "np.full(n, init_vals[{}], data[{}].dtype)".format(i, i)
                for i in range(count)
            ]
        )

    func_text = "def f(n, data, init_vals=()):\n"
    func_text += "  return ({}{})\n".format(
        allocs, "," if count == 1 else ""
    )  # single value needs comma to become tuple

    loc_vars = {}
    exec(func_text, {"empty_like_type": empty_like_type, "np": np}, loc_vars)
    alloc_impl = loc_vars["f"]
    return alloc_impl


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def tuple_to_scalar(n):
    if isinstance(n, types.BaseTuple):
        return lambda n: n[0]
    return lambda n: n


def alloc_type(n, t):  # pragma: no cover
    return np.empty(n, t.dtype)


@overload(alloc_type)
def overload_alloc_type(n, t):
    typ = t.instance_type if isinstance(t, types.TypeRef) else t

    if isinstance(typ, bodo.hiframes.pd_categorical_ext.CategoricalArray):
        from bodo.hiframes.pd_categorical_ext import fix_cat_array_type

        return lambda n, t: fix_cat_array_type(np.empty(n, t.dtype))

    if typ.dtype == bodo.hiframes.datetime_date_ext.datetime_date_type:
        return lambda n, t: bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(n)

    dtype = numba.numpy_support.as_dtype(typ.dtype)

    # nullable int array
    if isinstance(typ, IntegerArrayType):
        return lambda n, t: bodo.libs.int_arr_ext.init_integer_array(
            np.empty(n, dtype),
            # XXX using full since nulls are not supported in shuffle keys
            np.full((tuple_to_scalar(n) + 7) >> 3, 255, np.uint8),
        )

    # nullable bool array
    if typ == boolean_array:
        return lambda n, t: bodo.libs.bool_arr_ext.init_bool_array(
            np.empty(n, np.bool_),
            # XXX using full since nulls are not supported in shuffle keys
            np.full((tuple_to_scalar(n) + 7) >> 3, 255, np.uint8),
        )

    # TODO: categorical needs fixing?
    # fix_cat_array_type(np.empty(n_out, arr.dtype))

    return lambda n, t: np.empty(n, dtype)


def full_type(n, val, t):  # pragma: no cover
    return np.full(n, val, t.dtype)


@overload(full_type)
def overload_full_type(n, val, t):
    typ = t.instance_type if isinstance(t, types.TypeRef) else t
    dtype = numba.numpy_support.as_dtype(typ.dtype)

    # nullable int array
    if isinstance(typ, IntegerArrayType):
        return lambda n, val, t: bodo.libs.int_arr_ext.init_integer_array(
            np.full(n, val, dtype),
            np.full((tuple_to_scalar(n) + 7) >> 3, 255, np.uint8),
        )

    # nullable bool array
    if typ == boolean_array:
        return lambda n, val, t: bodo.libs.bool_arr_ext.init_bool_array(
            np.full(n, val, np.bool_),
            np.full((tuple_to_scalar(n) + 7) >> 3, 255, np.uint8),
        )

    # TODO: categorical needs fixing?
    # fix_cat_array_type(np.full(n_send, val, arr.dtype))

    return lambda n, val, t: np.full(n, val, dtype)


@intrinsic
def get_ctypes_ptr(typingctx, ctypes_typ=None):
    assert isinstance(ctypes_typ, types.ArrayCTypes)

    def codegen(context, builder, sig, args):
        in_carr, = args
        ctinfo = context.make_helper(builder, sig.args[0], in_carr)
        return ctinfo.data

    return types.voidptr(ctypes_typ), codegen


@intrinsic
def incref(typingctx, data=None):
    """manual incref of data to workaround bugs. Should be avoided if possible.
    """

    def codegen(context, builder, signature, args):
        data_val, = args

        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)

    return types.void(data), codegen


def remove_return_from_block(last_block):
    # remove const none, cast, return nodes
    assert isinstance(last_block.body[-1], ir.Return)
    last_block.body.pop()
    assert (
        isinstance(last_block.body[-1], ir.Assign)
        and isinstance(last_block.body[-1].value, ir.Expr)
        and last_block.body[-1].value.op == "cast"
    )
    last_block.body.pop()
    if (
        isinstance(last_block.body[-1], ir.Assign)
        and isinstance(last_block.body[-1].value, ir.Const)
        and last_block.body[-1].value.value is None
    ):
        last_block.body.pop()


def include_new_blocks(
    blocks,
    new_blocks,
    label,
    new_body,
    remove_non_return=True,
    work_list=None,
    func_ir=None,
):
    inner_blocks = add_offset_to_labels(new_blocks, ir_utils._max_label + 1)
    blocks.update(inner_blocks)
    ir_utils._max_label = max(blocks.keys())
    scope = blocks[label].scope
    loc = blocks[label].loc
    inner_topo_order = find_topo_order(inner_blocks)
    inner_first_label = inner_topo_order[0]
    inner_last_label = inner_topo_order[-1]
    if remove_non_return:
        remove_return_from_block(inner_blocks[inner_last_label])
    new_body.append(ir.Jump(inner_first_label, loc))
    blocks[label].body = new_body
    label = ir_utils.next_label()
    blocks[label] = ir.Block(scope, loc)
    if remove_non_return:
        inner_blocks[inner_last_label].body.append(ir.Jump(label, loc))
    # new_body.clear()
    if work_list is not None:
        topo_order = find_topo_order(inner_blocks)
        for _label in topo_order:
            block = inner_blocks[_label]
            block.scope = scope
            numba.inline_closurecall._add_definitions(func_ir, block)
            work_list.append((_label, block))
    return label


def gen_getitem(out_var, in_var, ind, calltypes, nodes):
    loc = out_var.loc
    getitem = ir.Expr.static_getitem(in_var, ind, None, loc)
    calltypes[getitem] = None
    nodes.append(ir.Assign(getitem, out_var, loc))


def is_static_getsetitem(node):
    return is_expr(node, "static_getitem") or isinstance(node, ir.StaticSetItem)


def get_getsetitem_index_var(node, typemap, nodes):
    # node is either getitem/static_getitem expr or Setitem/StaticSetitem
    index_var = node.index_var if is_static_getsetitem(node) else node.index
    # sometimes index_var is None, so fix it
    # TODO: get rid of static_getitem in general
    if index_var is None:
        # TODO: test this path
        assert is_static_getsetitem(node)
        # literal type is preferred for uniform/easier getitem index match
        try:
            index_typ = types.literal(node.index)
        except:
            index_typ = numba.typeof(node.index)
        index_var = ir.Var(
            node.value.scope, ir_utils.mk_unique_var("dummy_index"), node.loc
        )
        typemap[index_var.name] = index_typ
        # TODO: can every const index be ir.Const?
        nodes.append(ir.Assign(ir.Const(node.index, node.loc), index_var, node.loc))
    return index_var


# don't copy value since it can fail
# for example, deepcopy in get_parfor_reductions can fail for ObjModeLiftedWith const
import copy

ir.Const.__deepcopy__ = lambda self, memo: ir.Const(self.value, copy.deepcopy(self.loc))


def is_call_assign(stmt):
    return (
        isinstance(stmt, ir.Assign)
        and isinstance(stmt.value, ir.Expr)
        and stmt.value.op == "call"
    )


def is_call(expr):
    return isinstance(expr, ir.Expr) and expr.op == "call"


def is_var_assign(inst):
    return isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var)


def is_assign(inst):
    return isinstance(inst, ir.Assign)


def is_expr(val, op):
    return isinstance(val, ir.Expr) and val.op == op


def sanitize_varname(varname):
    new_name = (
        varname.replace("$", "_").replace(".", "_").replace(":", "_").replace(" ", "_")
    )
    if not new_name[0].isalpha():
        new_name = "_" + new_name
    if not new_name.isidentifier() or keyword.iskeyword(new_name):
        new_name = mk_unique_var("new_name").replace(".", "_")
    return new_name


def dump_node_list(node_list):
    for n in node_list:
        print("   ", n)


def debug_prints():
    return numba.config.DEBUG_ARRAY_OPT == 1
