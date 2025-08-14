"""Implementations for distributed operators. Loaded as needed to reduce import time."""

import numpy as np
from numba.core import types

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.time_ext import TimeArrayType
from bodo.libs.array_item_arr_ext import (
    offset_type,
)
from bodo.libs.binary_arr_ext import binary_array_type
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.pd_datetime_arr_ext import DatetimeArrayType
from bodo.libs.str_arr_ext import (
    string_array_type,
)
from bodo.utils.typing import (
    BodoError,
)
from bodo.utils.utils import (
    numba_to_c_type,
)


def irecv_impl(arr, size, pe, tag, cond):
    """Implementation for distributed_api.irecv()"""
    from bodo.libs.distributed_api import get_type_enum, mpi_req_numba_type

    _irecv = types.ExternalFunction(
        "dist_irecv",
        mpi_req_numba_type(
            types.voidptr,
            types.int32,
            types.int32,
            types.int32,
            types.int32,
            types.bool_,
        ),
    )

    # Numpy array
    if isinstance(arr, types.Array):

        def impl(arr, size, pe, tag, cond=True):  # pragma: no cover
            type_enum = get_type_enum(arr)
            return _irecv(arr.ctypes, size, type_enum, pe, tag, cond)

        return impl

    # Primitive array
    if isinstance(arr, bodo.libs.primitive_arr_ext.PrimitiveArrayType):

        def impl(arr, size, pe, tag, cond=True):  # pragma: no cover
            np_arr = bodo.libs.primitive_arr_ext.primitive_to_np(arr)
            type_enum = get_type_enum(np_arr)
            return _irecv(np_arr.ctypes, size, type_enum, pe, tag, cond)

        return impl

    if arr == boolean_array_type:
        # Nullable booleans need their own implementation because the
        # data array stores 1 bit per boolean. As a result, the data array
        # requires separate handling.
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_bool(arr, size, pe, tag, cond=True):  # pragma: no cover
            n_bytes = (size + 7) >> 3
            data_req = _irecv(arr._data.ctypes, n_bytes, char_typ_enum, pe, tag, cond)
            null_req = _irecv(
                arr._null_bitmap.ctypes, n_bytes, char_typ_enum, pe, tag, cond
            )
            return (data_req, null_req)

        return impl_bool

    # nullable arrays
    if (
        isinstance(
            arr,
            (
                IntegerArrayType,
                FloatingArrayType,
                DecimalArrayType,
                TimeArrayType,
                DatetimeArrayType,
            ),
        )
        or arr == datetime_date_array_type
    ):
        # return a tuple of requests for data and null arrays
        type_enum = np.int32(numba_to_c_type(arr.dtype))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_nullable(arr, size, pe, tag, cond=True):  # pragma: no cover
            n_bytes = (size + 7) >> 3
            data_req = _irecv(arr._data.ctypes, size, type_enum, pe, tag, cond)
            null_req = _irecv(
                arr._null_bitmap.ctypes, n_bytes, char_typ_enum, pe, tag, cond
            )
            return (data_req, null_req)

        return impl_nullable

    # string arrays
    if arr in [binary_array_type, string_array_type]:
        offset_typ_enum = np.int32(numba_to_c_type(offset_type))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        # using blocking communication for string arrays instead since the array
        # slice passed in shift() may not stay alive (not a view of the original array)
        if arr == binary_array_type:
            alloc_fn = "bodo.libs.binary_arr_ext.pre_alloc_binary_array"
        else:
            alloc_fn = "bodo.libs.str_arr_ext.pre_alloc_string_array"
        func_text = f"""def impl(arr, size, pe, tag, cond=True):
            # recv the number of string characters and resize buffer to proper size
            n_chars = bodo.libs.distributed_api.recv(np.int64, pe, tag - 1)
            new_arr = {alloc_fn}(size, n_chars)
            bodo.libs.str_arr_ext.move_str_binary_arr_payload(arr, new_arr)

            n_bytes = (size + 7) >> 3
            bodo.libs.distributed_api._recv(
                bodo.libs.str_arr_ext.get_offset_ptr(arr),
                size + 1,
                offset_typ_enum,
                pe,
                tag,
            )
            bodo.libs.distributed_api._recv(
                bodo.libs.str_arr_ext.get_data_ptr(arr), n_chars, char_typ_enum, pe, tag
            )
            bodo.libs.distributed_api._recv(
                bodo.libs.str_arr_ext.get_null_bitmap_ptr(arr),
                n_bytes,
                char_typ_enum,
                pe,
                tag,
            )
            return None"""

        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "np": np,
                "offset_typ_enum": offset_typ_enum,
                "char_typ_enum": char_typ_enum,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    raise BodoError(f"irecv(): array type {arr} not supported yet")
