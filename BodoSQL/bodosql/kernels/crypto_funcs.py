"""
Implements wrappers to call the C++ BodoSQL array kernels for SHA2 and other crypto functions.
"""

import llvmlite.binding as ll
import numba
import numpy as np
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic, overload

import bodo
from bodo.libs import crypto_funcs
from bodo.utils.typing import get_overload_const_bool

ll.add_symbol(
    "run_crypto_function",
    crypto_funcs.run_crypto_function,
)

ll.add_symbol(
    "run_base64_encode",
    crypto_funcs.run_base64_encode,
)

ll.add_symbol(
    "run_base64_decode_string",
    crypto_funcs.run_base64_decode_string,
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
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="run_crypto_function"
        )
        return builder.call(
            fn,
            [
                in_str_struct.data,
                in_str_struct.length,
                crypto_func,
                out_str_struct.data,
            ],
        )

    return types.void(in_str_typ, crypto_func_typ, out_str_typ), codegen


def sha2_algorithms(msg, digest_size):  # pragma: no cover
    """Function used to calculate the result of SHA encryption"""


@overload(sha2_algorithms)
def overload_sha2_algorithms(msg, digest_size):
    kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND

    def impl(msg, digest_size):  # pragma: no cover
        output = numba.cpython.unicode._empty_string(kind, digest_size // 4, 1)
        run_crypto_function(msg, np.int32(digest_size), output)
        return output

    return impl


def md5_algorithm(msg):  # pragma: no cover
    """Function used to calculate the result of MD5 encryption"""


@overload(md5_algorithm)
def overload_md5_algorithm(msg):
    kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND

    def impl(msg):  # pragma: no cover
        output = numba.cpython.unicode._empty_string(kind, 32, 1)
        run_crypto_function(msg, np.int32(0), output)
        return output

    return impl


@intrinsic
def run_hex_encode(
    typingctx,
    in_str_t,
    in_length_t,
    out_str_t,
):
    """Call C++ implementation of bytes_to_hex on a string input"""

    def codegen(context, builder, sig, args):
        in_str, in_length, out_str = args
        out_str_struct = cgutils.create_struct_proxy(types.unicode_type)(
            context, builder, value=out_str
        )
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
            ],
        )
        hex_func = cgutils.get_or_insert_function(
            builder.module, fnty, name="bytes_to_hex"
        )
        builder.call(hex_func, (out_str_struct.data, in_str, in_length))

    return (
        types.void(
            types.voidptr,
            in_length_t,
            out_str_t,
        ),
        codegen,
    )


def hex_encode_algorithm(
    msg, max_line_length, char_63, char_64, char_pad
):  # pragma: no cover
    """Function used to calculate the result of hex encryption"""


@overload(hex_encode_algorithm)
def overload_hex_encode_algorithm(msg):
    """
    Computes the result of the hex encoding algorithm on a scalar string.

    Args:
        msg (string): the string to be encoded

    Answer:
        (string): the hex-encoded string
    """
    kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND

    def impl(msg):  # pragma: no cover
        # Every 3 bytes in the input becomes 2 bytes in the output
        utf8_str, utf8_len = bodo.libs.str_ext.unicode_to_utf8_and_len(msg)
        output = numba.cpython.unicode._empty_string(kind, utf8_len * 2, 1)
        run_hex_encode(
            utf8_str,
            np.int64(utf8_len),
            output,
        )
        return output

    return impl


def hex_decode_string_algorithm(msg):  # pragma: no cover
    """Function used to calculate the result of hex decryption"""


def hex_decode_binary_algorithm(msg):  # pragma: no cover
    """Function used to calculate the result of hex decryption"""


@overload(hex_decode_string_algorithm)
def overload_hex_decode_string_algorithm(msg):  # pragma: no cover
    """
    Computes the result of the hex decoding algorithm on a scalar string.

    Args:
        msg (string): the string to be encoded

    Answer:
        (string, boolean): the hex-decoded string, and a boolean indicating success vs failure
    """

    def impl(msg):  # pragma: no cover
        if len(msg) == 0:
            return "", True

        # Verify that the number of characters is a multiple of 2
        if (len(msg) % 2) != 0:
            return "", False

        # Verify that all characters are hex characters
        for c in msg:
            if c not in "0123456789abcdefABCDEF":
                return "", False

        # Every 2 bytes in the encoded string corresponds to 1 byte in the decoded string
        decoded_length = len(msg) // 2
        output = np.array([0] * decoded_length, dtype=np.uint8)
        bodo.libs.binary_arr_ext._bytes_fromhex(output.ctypes, msg._data, len(msg))
        return bodo.libs.str_arr_ext.decode_utf8(output.ctypes, decoded_length), True

    return impl


@overload(hex_decode_binary_algorithm)
def overload_hex_decode_binary_algorithm(msg):  # pragma: no cover
    """
    Computes the result of the hex decoding algorithm on a scalar string.

    Args:
        msg (string): the string to be encoded

    Answer:
        (string, boolean): the hex-decoded binary data, and a boolean indicating success vs failure
    """

    def impl(msg):  # pragma: no cover
        if len(msg) == 0:
            return b"", True

        # Verify that the number of characters is a multiple of 2
        if (len(msg) % 2) != 0:
            return b"", False

        # Verify that all characters are hex characters
        for c in msg:
            if c not in "0123456789abcdefABCDEF":
                return b"", False

        # Every 2 bytes in the encoded string corresponds to 1 byte in the decoded string
        decoded_length = len(msg) // 2
        output = np.array([0] * decoded_length, dtype=np.uint8)
        bodo.libs.binary_arr_ext._bytes_fromhex(output.ctypes, msg._data, len(msg))
        return bodo.libs.binary_arr_ext.init_bytes_type(output, decoded_length), True

    return impl


@overload(hex_encode_algorithm)
def overload_hex_encode_algorithm(msg):
    """
    Computes the result of the hex encoding algorithm on a scalar string.

    Args:
        msg (string): the string to be encoded

    Answer:
        (string): the hex-encoded string
    """
    kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND

    def impl(msg):  # pragma: no cover
        # Every 3 bytes in the input becomes 2 bytes in the output
        utf8_str, utf8_len = bodo.libs.str_ext.unicode_to_utf8_and_len(msg)
        output = numba.cpython.unicode._empty_string(kind, utf8_len * 2, 1)
        run_hex_encode(
            utf8_str,
            np.int64(utf8_len),
            output,
        )
        return output

    return impl


@intrinsic
def run_base64_encode(
    typingctx,
    in_str_t,
    in_length_t,
    out_length_t,
    max_line_length_t,
    char_62_t,
    char_63_t,
    char_pad_t,
    out_str_t,
):
    """Call C++ implementation of run_base64_encode"""

    def codegen(context, builder, sig, args):
        (
            in_str,
            in_length,
            out_length,
            max_line_length,
            char_62,
            char_63,
            char_pad,
            out_str,
        ) = args
        out_str_struct = cgutils.create_struct_proxy(types.unicode_type)(
            context, builder, value=out_str
        )
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(32),
                lir.IntType(32),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="run_base64_encode"
        )
        return builder.call(
            fn,
            [
                in_str,
                in_length,
                out_length,
                max_line_length,
                char_62,
                char_63,
                char_pad,
                out_str_struct.data,
            ],
        )

    return (
        types.void(
            types.voidptr,
            in_length_t,
            out_length_t,
            max_line_length_t,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            out_str_t,
        ),
        codegen,
    )


def base64_encode_algorithm(
    msg, max_line_length, char_63, char_64, char_pad
):  # pragma: no cover
    """Function used to calculate the result of base64 encryption"""


@overload(base64_encode_algorithm)
def overload_base64_encode_algorithm(msg, max_line_length, char_63, char_64, char_pad):
    """
    Computes the result of the base64 encoding algorithm on a scalar string.

    Args:
        msg (string): the string to be encoded
        max_line_length (int): the number of characters to allow before injecting '\n'
        char_63 (string): a string containing the character to use instead of '+' for index 62
        char_64 (string): a string containing the character to use instead of '/' for index 63
        char_pad (string): a string containing the character to use instead of '=' for padding

    Answer:
        (string): the base64-encoded string
    """
    kind = numba.cpython.unicode.PY_UNICODE_1BYTE_KIND

    def impl(msg, max_line_length, char_63, char_64, char_pad):  # pragma: no cover
        # Every 3 bytes in the input becomes 4 bytes in the output (with padding)
        utf8_str, utf8_len = bodo.libs.str_ext.unicode_to_utf8_and_len(msg)
        out_length = ((utf8_len + 2) // 3) * 4
        if max_line_length > 0 and out_length > 0:
            out_length += out_length // max_line_length - int(
                (out_length % max_line_length) == 0
            )
        output = numba.cpython.unicode._empty_string(kind, out_length, 1)
        run_base64_encode(
            utf8_str,
            np.int32(utf8_len),
            np.int32(out_length),
            np.int32(max_line_length),
            bodo.libs.str_ext.unicode_to_utf8_and_len(char_63)[0],
            bodo.libs.str_ext.unicode_to_utf8_and_len(char_64)[0],
            bodo.libs.str_ext.unicode_to_utf8_and_len(char_pad)[0],
            output,
        )
        return output

    return impl


@intrinsic
def run_base64_decode_string(
    typingctx,
    in_str_t,
    in_length_t,
    char_62_t,
    char_63_t,
    char_pad_t,
    out_str_t,
):
    """Call C++ implementation of run_base64_decode_string"""

    def codegen(context, builder, sig, args):
        (
            in_str,
            in_length,
            char_62,
            char_63,
            char_pad,
            out_str,
        ) = args
        out_str_struct = out_str
        # out_str_struct = cgutils.create_struct_proxy(types.unicode_type)(
        #     context, builder, value=out_str
        # )
        fnty = lir.FunctionType(
            lir.IntType(1),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="run_base64_decode_string"
        )
        return builder.call(
            fn,
            [
                in_str,
                in_length,
                char_62,
                char_63,
                char_pad,
                # out_str_struct.data,
                out_str_struct,
            ],
        )

    return (
        types.bool_(
            types.voidptr,
            in_length_t,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.voidptr,
        ),
        codegen,
    )


def base64_decode_algorithm(
    msg, char_63, char_64, char_pad, _is_str
):  # pragma: no cover
    """Function used to calculate the result of base64 decryption"""


@overload(base64_decode_algorithm, no_unliteral=True)
def overload_base64_decode_algorithm(msg, char_63, char_64, char_pad, _is_str):
    """
    Computes the result of the base64 decoding algorithm on a scalar string.

    Args:
        msg (string): the string to be encoded
        char_63 (string): a string containing the character to use instead of '+' for index 62
        char_64 (string): a string containing the character to use instead of '/' for index 63
        char_pad (string): a string containing the character to use instead of '=' for padding

    Answer:
        (string, boolean): the base64-decoded string, and a boolean indicating success vs failure
    """
    _is_str_bool = get_overload_const_bool(_is_str)
    if _is_str_bool:

        def impl(msg, char_63, char_64, char_pad, _is_str):  # pragma: no cover
            if len(msg) == 0:
                return "", True
            # Remove newline characters, since they are injected without altering content
            msg_without_newline = msg.replace("\n", "")
            # Verify that there are at most 2 of the padding character, and
            # that they only occur at the end of the string.
            pad_count = msg_without_newline.count(char_pad)
            if pad_count > 2:
                return "", False
            trimmed = msg_without_newline.rstrip(char_pad)
            if trimmed.count(char_pad) != 0:
                return "", False
            # Every 4 bytes in the encoded string corresponds to 3 bytes in the decoded string
            # (ignoring padded characters)
            utf8_str, utf8_len = bodo.libs.str_ext.unicode_to_utf8_and_len(
                msg_without_newline
            )
            decoded_length = (utf8_len // 4) * 3 - pad_count
            output = np.array([0] * decoded_length, dtype=np.uint8)
            success = run_base64_decode_string(
                utf8_str,
                np.int32(utf8_len),
                bodo.libs.str_ext.unicode_to_utf8_and_len(char_63)[0],
                bodo.libs.str_ext.unicode_to_utf8_and_len(char_64)[0],
                bodo.libs.str_ext.unicode_to_utf8_and_len(char_pad)[0],
                output.ctypes,
            )
            return (
                bodo.libs.str_arr_ext.decode_utf8(output.ctypes, decoded_length),
                success,
            )

        return impl

    else:

        def impl(msg, char_63, char_64, char_pad, _is_str):  # pragma: no cover
            if len(msg) == 0:
                return b"", True
            # Remove newline characters, since they are injected without altering content
            msg_without_newline = msg.replace("\n", "")
            # Verify that there are at most 2 of the padding character, and
            # that they only occur at the end of the string.
            pad_count = msg_without_newline.count(char_pad)
            if pad_count > 2:
                return b"", False
            trimmed = msg_without_newline.rstrip(char_pad)
            if trimmed.count(char_pad) != 0:
                return b"", False
            utf8_str, utf8_len = bodo.libs.str_ext.unicode_to_utf8_and_len(
                msg_without_newline
            )
            # Every 4 bytes in the encoded string corresponds to 3 bytes in the decoded string
            # (ignoring padded characters)
            decoded_length = (utf8_len // 4) * 3 - pad_count
            output = np.array([0] * decoded_length, dtype=np.uint8)
            success = run_base64_decode_string(
                utf8_str,
                np.int32(utf8_len),
                bodo.libs.str_ext.unicode_to_utf8_and_len(char_63)[0],
                bodo.libs.str_ext.unicode_to_utf8_and_len(char_64)[0],
                bodo.libs.str_ext.unicode_to_utf8_and_len(char_pad)[0],
                output.ctypes,
            )
            return (
                bodo.libs.binary_arr_ext.init_bytes_type(output, decoded_length),
                success,
            )

        return impl
