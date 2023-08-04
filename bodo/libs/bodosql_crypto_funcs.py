# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Implements wrappers to call the C++ BodoSQL array kernels for SHA2 and other crypto functions.
"""
import types

import llvmlite.binding as ll

import bodo
from bodo.libs import crypto_funcs
from llvmlite import ir as lir
from numba.core import cgutils, types
import numba
from numba.extending import (
    intrinsic,
    overload,
)
import numpy as np

ll.add_symbol(
    "run_crypto_function",
    crypto_funcs.run_crypto_function,
)

_run_crypto_function = types.ExternalFunction(
    "run_crypto_function",
    types.void(
        types.voidptr,
        types.int32,
        types.int32,
        types.voidptr,
    ),
)


@intrinsic
def run_crypto_function(typingctx, in_str_typ, crypto_func_typ, out_str_typ):
    """Call C++ implementation of run_crypto_function"""
    def codegen(context, builder, sig, args):
        (in_str, crypto_func, out_str) = args
        in_str_struct = cgutils.create_struct_proxy(types.unicode_type)(
            context, builder, value=in_str
        )
        out_str_struct = cgutils.create_struct_proxy(types.unicode_type)(
            context, builder, value=out_str
        )
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(32),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(builder.module, fnty, name="run_crypto_function")
        return builder.call(
            fn,
            [
                in_str_struct.data,
                in_str_struct.length,
                crypto_func,
                out_str_struct.data,
            ]
        )


    return types.void(in_str_typ, crypto_func_typ, out_str_typ), codegen


def sha2_algorithms(msg, digest_size): # pragma: no cover
    """Function used to calculate the result of SHA encryption"""


@overload(sha2_algorithms)
def overload_sha2_algorithms(msg, digest_size):
    kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND
    def impl(msg, digest_size):  # pragma: no cover
        output = numba.cpython.unicode._empty_string(kind, digest_size // 4, 1)
        run_crypto_function(msg, np.int32(digest_size), output)
        return output

    return impl


def md5_algorithm(msg): # pragma: no cover
    """Function used to calculate the result of MD5 encryption"""


@overload(md5_algorithm)
def overload_md5_algorithm(msg):
    kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND
    def impl(msg):  # pragma: no cover
        output = numba.cpython.unicode._empty_string(kind, 32, 1)
        run_crypto_function(msg, np.int32(0), output)
        return output

    return impl
