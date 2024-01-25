# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Implements array kernels that are specific to the HASH function in BodoSQL
"""

import numpy as np
from numba import literal_unroll
from numba.core import types
from numba.cpython.hashing import (
    _Py_HASH_CUTOFF,
    _Py_uhash_t,
    _PyHASH_XXPRIME_1,
    _PyHASH_XXPRIME_2,
    _PyHASH_XXPRIME_5,
    _PyHASH_XXROTATE,
    float_hash,
    grab_byte,
    grab_uint64_t,
    int_hash,
    process_return,
)
from numba.cpython.unicode import _kind_to_byte_width
from numba.extending import overload, register_jitable

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import raise_bodo_error
from bodo.utils.utils import is_array_typ

"""
The following utilities were copied over from numba.cpython.hashing since
they cannot be directly imported
"""


@register_jitable
def _ROTATE(x, b):  # pragma: no cover
    return types.uint64(((x) << (b)) | ((x) >> (types.uint64(64) - (b))))


@register_jitable
def _HALF_ROUND(a, b, c, d, s, t):  # pragma: no cover
    a += b
    c += d
    b = _ROTATE(b, s) ^ a
    d = _ROTATE(d, t) ^ c
    a = _ROTATE(a, 32)
    return a, b, c, d


@register_jitable
def _SINGLE_ROUND(v0, v1, v2, v3):  # pragma: no cover
    v0, v1, v2, v3 = _HALF_ROUND(v0, v1, v2, v3, 13, 16)
    v2, v1, v0, v3 = _HALF_ROUND(v2, v1, v0, v3, 17, 21)
    return v0, v1, v2, v3


@register_jitable
def _DOUBLE_ROUND(v0, v1, v2, v3):  # pragma: no cover
    v0, v1, v2, v3 = _SINGLE_ROUND(v0, v1, v2, v3)
    v0, v1, v2, v3 = _SINGLE_ROUND(v0, v1, v2, v3)
    return v0, v1, v2, v3


"""
End of directly copied functions
"""


def sql_hash(A, scalars):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(sql_hash)
def overload_sql_hash(A, scalars):
    """Handles cases where HASH receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if not isinstance(A, (types.Tuple, types.UniTuple)):
        raise_bodo_error("Hash argument must be a tuple")
    for i in range(len(A)):
        if isinstance(A[i], types.optional):
            # Note: If we have an optional scalar and its not the last argument,
            # then the NULL vs non-NULL case can lead to different decisions
            # about dictionary encoding in the output. This will lead to a memory
            # leak as the dict-encoding result will be cast to a regular string array.
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.sql_hash",
                ["A", "scalars"],
                0,
                container_arg=i,
                container_length=len(A),
            )

    def impl(A, scalars):  # pragma: no cover
        return sql_hash_util(A, scalars)

    return impl


def bytes_hash(val, length, is_str):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(bytes_hash)
def overload_bytes_hash(val, length, is_str):
    """
    A modified version of _Py_HashBytes from the numba hashing implementation at
    https://github.com/numba/numba/blob/main/numba/cpython/hashing.py

    In the original, several "hash secrets" are randomly seeded at the start of a
    Python  process, thus ensuring that the hash function itself is deterministic
    within  a single Python process but random across multiple processes.

    This implementation hardcodes 3 such secrets to arbitrary values selected
    by examining their values during an arbitrary Python process. This allows us
    to have a truly deterministic function that still has all of the potent
    hashing properties that Python's builtin hash() function has.

    Also takes in an extra argument as a is_str which adjusts the secret values
    slightly so that identical strings and binary data have distinct hashes.
    This prevents "cat" and b"cat" from hashing to the same value every time.
    """

    def impl(val, length, is_str):  # pragma: no cover
        if is_str:
            djbx33a_suffix = types.uint64(64130519237923773)
            k0 = types.uint64(3616741503670771235)
            k1 = types.uint64(10739179702160894625)
        else:
            djbx33a_suffix = types.uint64(199160714788894)
            k0 = types.uint64(55188675050165)
            k1 = types.uint64(163869740457570)
        if length == 0:
            return process_return(0)

        if length < _Py_HASH_CUTOFF:
            _hash = _Py_uhash_t(5381)
            for idx in range(length):
                _hash = ((_hash << 5) + _hash) + np.uint8(grab_byte(val, idx))

            _hash ^= length
            _hash ^= djbx33a_suffix
        else:
            b = types.uint64(length) << 56
            v0 = k0 ^ types.uint64(0x736F6D6570736575)
            v1 = k1 ^ types.uint64(0x646F72616E646F6D)
            v2 = k0 ^ types.uint64(0x6C7967656E657261)
            v3 = k1 ^ types.uint64(0x7465646279746573)

            idx = 0
            while length >= 8:
                mi = grab_uint64_t(val, idx)
                idx += 1
                length -= 8
                v3 ^= mi
                v0, v1, v2, v3 = _DOUBLE_ROUND(v0, v1, v2, v3)
                v0 ^= mi
            t = types.uint64(0x0)
            boffset = idx * 8
            ohexefef = types.uint64(0xFF)
            if length >= 7:
                jmp = 6 * 8
                mask = ~types.uint64(ohexefef << jmp)
                t = (t & mask) | (types.uint64(grab_byte(val, boffset + 6)) << jmp)
            if length >= 6:
                jmp = 5 * 8
                mask = ~types.uint64(ohexefef << jmp)
                t = (t & mask) | (types.uint64(grab_byte(val, boffset + 5)) << jmp)
            if length >= 5:
                jmp = 4 * 8
                mask = ~types.uint64(ohexefef << jmp)
                t = (t & mask) | (types.uint64(grab_byte(val, boffset + 4)) << jmp)
            if length >= 4:
                t &= types.uint64(0xFFFFFFFF00000000)
                for i in range(4):
                    jmp = i * 8
                    mask = ~types.uint64(ohexefef << jmp)
                    t = (t & mask) | (types.uint64(grab_byte(val, boffset + i)) << jmp)
            if length >= 3:
                jmp = 2 * 8
                mask = ~types.uint64(ohexefef << jmp)
                t = (t & mask) | (types.uint64(grab_byte(val, boffset + 2)) << jmp)
            if length >= 2:
                jmp = 1 * 8
                mask = ~types.uint64(ohexefef << jmp)
                t = (t & mask) | (types.uint64(grab_byte(val, boffset + 1)) << jmp)
            if length >= 1:
                mask = ~(ohexefef)
                t = (t & mask) | (types.uint64(grab_byte(val, boffset + 0)))

            b |= t
            v3 ^= b
            v0, v1, v2, v3 = _DOUBLE_ROUND(v0, v1, v2, v3)
            v0 ^= b
            v2 ^= ohexefef
            v0, v1, v2, v3 = _DOUBLE_ROUND(v0, v1, v2, v3)
            v0, v1, v2, v3 = _DOUBLE_ROUND(v0, v1, v2, v3)
            _hash = (v0 ^ v1) ^ (v2 ^ v3)
        return process_return(_hash)

    return impl


def unicode_hash(val):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(unicode_hash)
def overload_unicode_hash(val):
    """
    A modified version of unicode_hash from the numba hashing implementation at
    https://github.com/numba/numba/blob/main/numba/cpython/hashing.py

    The modifications replace _PyHashBytes (which uses hash secrets) with
    bytes_hash, defined above.
    """

    def impl(val):  # pragma: no cover
        kindwidth = _kind_to_byte_width(val._kind)
        _len = len(val)
        return bytes_hash(val._data, kindwidth * _len, True)

    return impl


def tuple_hash(t):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(tuple_hash)
def overload_tuple_hash(tup):
    """
    A modified version of _tuple_hash from the numba hashing implementation at
    https://github.com/numba/numba/blob/main/numba/cpython/hashing.py

    This utility calls consistent_hash as a subroutine instead of bultin hash.
    """

    def impl(tup):  # pragma: no cover
        tl = len(tup)
        acc = _PyHASH_XXPRIME_5
        hash_negative_1 = _Py_uhash_t(-1)
        for x in literal_unroll(tup):
            lane = consistent_hash(x)
            if lane == hash_negative_1:
                return -1
            acc += lane * _PyHASH_XXPRIME_2
            acc = _PyHASH_XXROTATE(acc)
            acc *= _PyHASH_XXPRIME_1

        acc += tl ^ (_PyHASH_XXPRIME_5 ^ _Py_uhash_t(3527539))

        if acc == hash_negative_1:
            return process_return(1546275796)

        return process_return(acc)

    return impl


def array_hash(t):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(array_hash)
def overload_array_hash(arr):
    """
    A variant of tuple_hash for scalar arrays.
    """

    def impl(arr):  # pragma: no cover
        n = len(arr)
        acc = _PyHASH_XXPRIME_5
        hash_negative_1 = _Py_uhash_t(-1)
        for i in range(n):
            if bodo.libs.array_kernels.isna(arr, i):
                lane = -1
            else:
                lane = consistent_hash(arr[i])
            if lane == hash_negative_1:
                return -1
            acc += lane * _PyHASH_XXPRIME_2
            acc = _PyHASH_XXROTATE(acc)
            acc *= _PyHASH_XXPRIME_1

        acc += n ^ (_PyHASH_XXPRIME_5 ^ _Py_uhash_t(3527539))

        if acc == hash_negative_1:
            return process_return(1546275796)

        return process_return(acc)

    return impl


def struct_hash(struct):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(struct_hash)
def overload_struct_hash(struct):
    """
    Handler for consistent_hash on structs: hashes each field,
    hashes all the names, then hashes a tuple of all the hashes combined.
    """

    args = []
    for idx, name in enumerate(struct.names):
        args.append(f"consistent_hash({repr(name)})")
        args.append(
            f"consistent_hash(get_struct_data(struct)[{idx}]) if get_struct_null_bitmap(struct)[{idx}] else -1"
        )
    func_text = "def impl(struct):\n"
    func_text += f"  return consistent_hash(({', '.join(args)}))\n"

    loc_vars = {}
    exec(
        func_text,
        {
            "consistent_hash": consistent_hash,
            "get_struct_data": bodo.libs.struct_arr_ext.get_struct_data,
            "get_struct_null_bitmap": bodo.libs.struct_arr_ext.get_struct_null_bitmap,
        },
        loc_vars,
    )
    return loc_vars["impl"]


def map_hash(map_scalar):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(map_hash)
def overload_map_hash(map_scalar):
    """
    Handler for consistent_hash on map scalars: hashes the
    key array, the value array, then hashes the tuple of
    the two combined.
    """

    def impl(map_scalar):  # pragma: no cover
        key_hash = consistent_hash(map_scalar._keys)
        val_hash = consistent_hash(map_scalar._values)
        return consistent_hash((key_hash, val_hash))

    return impl


def consistent_hash(val):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(consistent_hash)
def overload_consistent_hash(val):  # pragma: no cover
    """
    A hash function for BodoSQL scalar values that is deterministic so that its
    outputs are consistent across multiple ranks or multiple sessions.

    Args:
        val: the value being hashed which can currently be one of the following
        supported types:
        - Integer
        - Float
        - Boolean
        - String
        - Binary
        - Date
        - Timestamp (with or without a timezone)
        - Time
        - Decimal
        - A tuple, struct, map scalar, or array of values of any of the types above

    Returns:
        int64: a deterministic hash value for val with all of the expected
        properties of a hash function.
    """
    if bodo.utils.utils.is_array_typ(val, False):

        def impl(val):  # pragma: no cover
            return array_hash(val)

    elif is_valid_int_arg(val):

        def impl(val):  # pragma: no cover
            return int_hash(val)

    elif is_valid_boolean_arg(val):

        def impl(val):  # pragma: no cover
            return int_hash(int(val))

    elif is_valid_float_arg(val):

        def impl(val):  # pragma: no cover
            return float_hash(val)

    elif is_valid_string_arg(val):

        def impl(val):  # pragma: no cover
            return unicode_hash(val)

    elif is_valid_binary_arg(val):

        def impl(val):  # pragma: no cover
            return bytes_hash(val._data, len(val), False)

    elif is_valid_date_arg(val):

        def impl(val):  # pragma: no cover
            return int_hash(
                bodo.hiframes.datetime_date_ext.cast_datetime_date_to_int(val)
            )

    elif is_valid_time_arg(val):

        def impl(val):  # pragma: no cover
            return int_hash(val.value)

    elif is_valid_tz_naive_datetime_arg(val):

        def impl(val):  # pragma: no cover
            return int_hash(bodo.utils.conversion.unbox_if_tz_naive_timestamp(val))

    elif is_valid_tz_aware_datetime_arg(val):

        def impl(val):  # pragma: no cover
            return int_hash(val.value)

    elif isinstance(val, bodo.Decimal128Type):

        def impl(val):  # pragma: no cover
            return consistent_hash(str(val))

    elif isinstance(val, bodo.libs.struct_arr_ext.StructType):

        def impl(val):  # pragma: no cover
            return struct_hash(val)

    elif isinstance(val, bodo.libs.map_arr_ext.MapScalarType):

        def impl(val):  # pragma: no cover
            return map_hash(val)

    elif isinstance(val, types.UniTuple):

        def impl(val):  # pragma: no cover
            return tuple_hash(val)

    else:
        raise_bodo_error(f"Unsupported type for sql_hash: {val}")
    return impl


def sql_hash_util(A):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(sql_hash_util, no_unliteral=True)
def overload_sql_hash_util(A, scalars):
    """A dedicated kernel for the SQL function HASH which takes in 1+ values
       of any type and outputs a hash produced by their combined values such
       that NULL is treated as a distinct value in its own right.

    Args:
        A (any array/scalar tuple): the tuple of values to be hashed together
        scalars (boolean tuple): which of the arguments are scalars

    Raises:
        BodoError: if there are 0 columns

    Returns:
        an array containing the result of hashing each row of the tuple of
        columns

    Note: for each row of the inputs columns, sql_hash will hash each non-null
    value, treat NULLs as hash-value -1, place all the individual hashes in
    a tuple and then hash the tuple. This is so that NULL values can be treated
    as a distinct value in their own right without an obvious collision. The
    value -1 is chosen because 0 and 1 are the hash values for True and False,
    and no obvious integers hash to -1.
    """

    if len(A) == 0:  # pragma: no cover
        raise_bodo_error("Cannot hash 0 columns")

    scalars = bodo.utils.typing.unwrap_typeref(scalars).meta
    are_arrays = [not scalars[i] for i in range(len(scalars))]

    arg_names = []
    arg_types = []

    for i, arr_typ in enumerate(A):
        arg_name = f"A{i}"
        arg_names.append(arg_name)
        arg_types.append(arr_typ)

    propagate_null = [False] * len(arg_names)

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int64)

    # Create the mapping from the tuple to the local variable.
    arg_string = "A, scalars"
    arg_sources = {f"A{i}": f"A[{i}]" for i in range(len(A))}

    scalar_text = ""
    hash_vals = []
    for i, typ in enumerate(arg_types):
        if typ == bodo.none:
            val = "-1"
        else:
            val = f"h{i}"
            hash_arg = f"consistent_hash(arg{i})"
            if is_array_typ(typ):
                hash_arg = (
                    f"-1 if bodo.libs.array_kernels.isna(A{i}, i) else {hash_arg}"
                )
            scalar_text += f"{val} = {hash_arg}\n"
        hash_vals.append(val)
    scalar_text += f"res[i] = consistent_hash(({', '.join(hash_vals)},))"

    extra_globals = {"consistent_hash": consistent_hash}

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        arg_string,
        arg_sources,
        extra_globals=extra_globals,
        are_arrays=are_arrays,
    )
