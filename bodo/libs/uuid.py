# Copyright (C) 2024 Bodo Inc. All rights reserved.

import llvmlite.binding as ll
import numba
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic, overload

import bodo.utils.utils
from bodo.ext import uuid_cpp

ll.add_symbol("uuidV4", uuid_cpp.uuidV4)
ll.add_symbol("uuidV5", uuid_cpp.uuidV5)


string_type = types.unicode_type


@intrinsic
def uuidV4_wrapper(typingctx, output):
    """Wrapper for uuidV4 in _uuid.cpp"""

    def codegen(context, builder, sig, args):
        output = args[0]
        output_struct = cgutils.create_struct_proxy(string_type)(
            context, builder, value=output
        )

        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
        fn_typ = cgutils.get_or_insert_function(builder.module, fnty, name="uuidV4")
        builder.call(fn_typ, (output_struct.data,))
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(output)
    return sig, codegen


@intrinsic
def uuidV5_wrapper(typingctx, output, namespace, name):
    """Wrapper for uuidV5 in _uuid.cpp"""

    def codegen(context, builder, sig, args):
        output, namespace, name = args
        output_struct = cgutils.create_struct_proxy(string_type)(
            context, builder, value=output
        )

        ns_struct = cgutils.create_struct_proxy(string_type)(
            context, builder, value=namespace
        )
        name_struct = cgutils.create_struct_proxy(string_type)(
            context, builder, value=name
        )

        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
            ],
        )
        fn_typ = cgutils.get_or_insert_function(builder.module, fnty, name="uuidV5")
        builder.call(
            fn_typ,
            (
                output_struct.data,
                ns_struct.data,
                ns_struct.length,
                name_struct.data,
                name_struct.length,
            ),
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    sig = types.void(output, namespace, name)
    return sig, codegen


def uuidV4():  # pragma: no cover
    """Bodo implementation of UUIDv4"""
    pass


@overload(uuidV4)
def overload_uuidV4():
    kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND

    def impl():  # pragma: no cover
        output = numba.cpython.unicode._empty_string(kind, 36, 1)
        uuidV4_wrapper(output)
        return output

    return impl


def uuidV5(namespace, name):  # pragma: no cover
    """Bodo implementation of UUIDv5. If the namespace is ill-formed, returns
    the empty string"""
    pass


@overload(uuidV5)
def overload_uuidV5(namespace, name):
    kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND

    def impl(namespace, name):  # pragma: no cover
        output = numba.cpython.unicode._empty_string(kind, 36, 1)
        uuidV5_wrapper(output, namespace, name)
        if output[0] == "\x00":
            return ""
        return output

    return impl
