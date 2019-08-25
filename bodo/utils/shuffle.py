"""
helper data structures and functions for shuffle (alltoall).
"""

from collections import namedtuple
import numpy as np

import numba
from numba import types
from numba.extending import overload

import bodo
from bodo.utils.utils import get_ctypes_ptr, _numba_to_c_type_map
from bodo.libs.timsort import getitem_arr_tup
from bodo.libs.str_ext import string_type
from bodo.libs.str_arr_ext import (string_array_type, to_string_list,
    get_offset_ptr, get_data_ptr, convert_len_arr_to_offset, set_bit_to,
    pre_alloc_string_array, num_total_chars, get_null_bitmap_ptr,
    get_bit_bitmap)
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import BooleanArrayType, boolean_array


########## metadata required for shuffle
# send_counts -> pre, single
# recv_counts -> single
# send_buff
# out_arr
# n_send  -> single
# n_out  -> single
# send_disp -> single
# recv_disp -> single
# send_disp_nulls -> single
# recv_disp_nulls -> single
# tmp_offset -> single
############### string arrays
# send_counts_char -> pre
# recv_counts_char
# send_arr_lens -> pre
# send_arr_nulls -> pre
# send_arr_chars
# send_disp_char
# recv_disp_char
# tmp_offset_char
#### dummy array to key reference count alive, since ArrayCTypes can't be
#### passed to jitclass TODO: update
# send_arr_chars_arr


PreShuffleMeta = namedtuple('PreShuffleMeta',
    'send_counts, send_counts_char_tup, send_arr_lens_tup, send_arr_nulls_tup')


ShuffleMeta = namedtuple('ShuffleMeta',
    ('send_counts, recv_counts, n_send, n_out, send_disp, recv_disp, '
    'send_disp_nulls, recv_disp_nulls, tmp_offset, send_buff_tup, out_arr_tup, '
    'send_counts_char_tup, recv_counts_char_tup, send_arr_lens_tup, '
    'send_arr_nulls_tup, send_arr_chars_tup, send_disp_char_tup, '
    'recv_disp_char_tup, tmp_offset_char_tup, send_arr_chars_arr_tup'))


# before shuffle, 'send_counts' is needed as well as
# 'send_counts_char', 'send_arr_lens' and 'send_arr_nulls' for every string
def alloc_pre_shuffle_metadata(arr, data, n_pes, is_contig):
    return PreShuffleMeta(np.zeros(n_pes, np.int32), ())


@overload(alloc_pre_shuffle_metadata)
def alloc_pre_shuffle_metadata_overload(key_arrs, data, n_pes, is_contig):

    func_text = "def f(key_arrs, data, n_pes, is_contig):\n"
    # send_counts
    func_text += "  send_counts = np.zeros(n_pes, np.int32)\n"

    # send_counts_char, send_arr_lens for strings
    n_keys = len(key_arrs.types)
    n_all = n_keys + len(data.types)
    for i, typ in enumerate(key_arrs.types + data.types):
        func_text += ("  arr = key_arrs[{}]\n".format(i) if i < n_keys
                      else "  arr = data[{}]\n".format(i - n_keys))
        if typ == string_array_type:
            func_text += "  send_counts_char_{} = np.zeros(n_pes, np.int32)\n".format(i)
            func_text += "  send_arr_lens_{} = np.empty(1, np.uint32)\n".format(i)
            # needs allocation since written in update before finalize
            func_text += "  if is_contig:\n"
            func_text += "    send_arr_lens_{} = np.empty(len(arr), np.uint32)\n".format(i)
        else:
            func_text += "  send_counts_char_{} = None\n".format(i)
            func_text += "  send_arr_lens_{} = None\n".format(i)

        # null masks for string array, int array
        if is_null_masked_type(typ):
            func_text += "  send_arr_nulls_{} = np.empty(1, np.uint8)\n".format(i)
            # allocate null bytes, 2 * n_pes extra space since bits are
            # unpacked around processor border
            func_text += "  if is_contig:\n"
            func_text += "    n_bytes = (len(arr) + 7) >> 3\n"
            # using full since NA in keys is not updated
            func_text += "    send_arr_nulls_{} = np.full(n_bytes + 2 * n_pes, 255, np.uint8)\n".format(i)
        else:
            func_text += "  send_arr_nulls_{} = None\n".format(i)

    count_char_tup = ", ".join("send_counts_char_{}".format(i)
                                                        for i in range(n_all))
    lens_tup = ", ".join("send_arr_lens_{}".format(i) for i in range(n_all))
    nulls_tup = ", ".join("send_arr_nulls_{}".format(i) for i in range(n_all))
    extra_comma = "," if n_all == 1 else ""
    func_text += "  return PreShuffleMeta(send_counts, ({}{}), ({}{}), ({}{}))\n".format(
        count_char_tup, extra_comma, lens_tup, extra_comma, nulls_tup,
        extra_comma)

    # print(func_text)
    loc_vars = {}
    exec(func_text, {'np': np, 'PreShuffleMeta': PreShuffleMeta}, loc_vars)
    alloc_impl = loc_vars['f']
    return alloc_impl


# 'send_counts' is updated, and 'send_counts_char' and 'send_arr_lens'
# for every string type
def update_shuffle_meta(pre_shuffle_meta, node_id, ind, val, data, is_contig=True, padded_bits=0):
    pre_shuffle_meta.send_counts[node_id] += 1


@overload(update_shuffle_meta)
def update_shuffle_meta_overload(pre_shuffle_meta, node_id, ind, val, data, is_contig=True, padded_bits=0):
    func_text = "def f(pre_shuffle_meta, node_id, ind, val, data, is_contig=True, padded_bits=0):\n"
    func_text += "  pre_shuffle_meta.send_counts[node_id] += 1\n"
    n_keys = len(val.types)
    for i, typ in enumerate(val.types + data.types):
        if typ in (string_type, string_array_type):
            val_or_data = 'val[{}]'.format(i) if i < n_keys else 'getitem_arr_tup(data, ind)[{}]'.format(i - n_keys)
            func_text += "  n_chars = len({})\n".format(val_or_data)
            func_text += "  pre_shuffle_meta.send_counts_char_tup[{}][node_id] += n_chars\n".format(i)
            func_text += "  if is_contig:\n"
            func_text += "    pre_shuffle_meta.send_arr_lens_tup[{}][ind] = n_chars\n".format(i)

        # XXX: handling null bits is only supported for data arrays now
        if i >= n_keys and is_null_masked_type(typ):
            func_text += "  if is_contig:\n"
            func_text += "    out_bitmap = pre_shuffle_meta.send_arr_nulls_tup[{}].ctypes\n".format(i)
            #func_text += "    bit_val = get_bit_bitmap(get_null_bitmap_ptr(data[{}]), ind)\n".format(i - n_keys)
            func_text += "    bit_val = get_mask_bit(data[{}], ind)\n".format(i - n_keys)
            func_text += "    set_bit_to(out_bitmap, padded_bits + ind, bit_val)\n"

    # print(func_text)
    loc_vars = {}
    exec(func_text, {'set_bit_to': set_bit_to,
        'get_bit_bitmap': get_bit_bitmap,
        'get_null_bitmap_ptr': get_null_bitmap_ptr,
        'getitem_arr_tup': getitem_arr_tup,
        'get_mask_bit': get_mask_bit}, loc_vars)
    update_impl = loc_vars['f']
    return update_impl


@numba.njit
def calc_disp_nulls(arr):
    disp = np.empty_like(arr)
    disp[0] = 0
    for i in range(1, len(arr)):
        l = (arr[i-1] + 7) >> 3
        disp[i] = disp[i-1] + l
    return disp


def finalize_shuffle_meta(arrs, data, pre_shuffle_meta, n_pes, is_contig, init_vals=()):
    return ShuffleMeta()


@overload(finalize_shuffle_meta)
def finalize_shuffle_meta_overload(key_arrs, data, pre_shuffle_meta, n_pes, is_contig, init_vals=()):

    func_text = "def f(key_arrs, data, pre_shuffle_meta, n_pes, is_contig, init_vals=()):\n"
    # common metas: send_counts, recv_counts, tmp_offset, n_out, n_send, send_disp, recv_disp
    func_text += "  send_counts = pre_shuffle_meta.send_counts\n"
    func_text += "  recv_counts = np.empty(n_pes, np.int32)\n"
    func_text += "  tmp_offset = np.zeros(n_pes, np.int32)\n"  # for non-contig
    func_text += "  bodo.libs.distributed_api.alltoall(send_counts, recv_counts, 1)\n"
    func_text += "  n_out = recv_counts.sum()\n"
    func_text += "  n_send = send_counts.sum()\n"
    func_text += "  send_disp = bodo.ir.join.calc_disp(send_counts)\n"
    func_text += "  recv_disp = bodo.ir.join.calc_disp(recv_counts)\n"
    func_text += "  send_disp_nulls = calc_disp_nulls(send_counts)\n"
    func_text += "  recv_disp_nulls = calc_disp_nulls(recv_counts)\n"

    n_keys = len(key_arrs.types)
    n_all = len(key_arrs.types + data.types)

    for i, typ in enumerate(key_arrs.types + data.types):
        func_text += ("  arr = key_arrs[{}]\n".format(i) if i < n_keys
                      else "  arr = data[{}]\n".format(i - n_keys))
        if typ == string_array_type:
            # send_buff is None for strings
            func_text += "  send_buff_{} = None\n".format(i)
            # send/recv counts
            func_text += "  send_counts_char_{} = pre_shuffle_meta.send_counts_char_tup[{}]\n".format(i, i)
            func_text += "  recv_counts_char_{} = np.empty(n_pes, np.int32)\n".format(i)
            func_text += ("  bodo.libs.distributed_api.alltoall("
                "send_counts_char_{}, recv_counts_char_{}, 1)\n").format(i, i)
            # alloc output
            func_text += "  n_all_chars = recv_counts_char_{}.sum()\n".format(i)
            func_text += "  out_arr_{} = pre_alloc_string_array(n_out, n_all_chars)\n".format(i)
            # send/recv disp
            func_text += ("  send_disp_char_{} = bodo.ir.join."
                "calc_disp(send_counts_char_{})\n").format(i, i)
            func_text += ("  recv_disp_char_{} = bodo.ir.join."
                "calc_disp(recv_counts_char_{})\n").format(i, i)

            # tmp_offset_char, send_arr_lens
            func_text += "  tmp_offset_char_{} = np.zeros(n_pes, np.int32)\n".format(i)
            func_text += "  send_arr_lens_{} = pre_shuffle_meta.send_arr_lens_tup[{}]\n".format(i, i)
            # send char arr
            # TODO: arr refcount if arr is not stored somewhere?
            func_text += "  send_arr_chars_arr_{} = np.empty(1, np.uint8)\n".format(i)
            func_text += "  send_arr_chars_{} = get_ctypes_ptr(get_data_ptr(arr))\n".format(i)
            func_text += "  if not is_contig:\n"
            func_text += "    send_arr_lens_{} = np.empty(n_send, np.uint32)\n".format(i)
            func_text += "    s_n_all_chars = send_counts_char_{}.sum()\n".format(i)
            func_text += "    send_arr_chars_arr_{} = np.empty(s_n_all_chars, np.uint8)\n".format(i)
            func_text += "    send_arr_chars_{} = get_ctypes_ptr(send_arr_chars_arr_{}.ctypes)\n".format(i, i)
        else:
            assert isinstance(typ, (types.Array, IntegerArrayType, BooleanArrayType))
            func_text += "  out_arr_{} = bodo.utils.utils.alloc_type(n_out, arr)\n".format(i)
            func_text += "  send_buff_{} = arr\n".format(i)
            func_text += "  if not is_contig:\n"
            if i >= n_keys and init_vals != ():
                func_text += "    send_buff_{} = bodo.utils.utils.full_type(n_send, init_vals[{}], arr)\n".format(i, i - n_keys)
            else:
                func_text += "    send_buff_{} = bodo.utils.utils.alloc_type(n_send, arr)\n".format(i)
            # string buffers are None
            func_text += "  send_counts_char_{} = None\n".format(i)
            func_text += "  recv_counts_char_{} = None\n".format(i)
            func_text += "  send_arr_lens_{} = None\n".format(i)
            func_text += "  send_arr_chars_{} = None\n".format(i)
            func_text += "  send_disp_char_{} = None\n".format(i)
            func_text += "  recv_disp_char_{} = None\n".format(i)
            func_text += "  tmp_offset_char_{} = None\n".format(i)
            func_text += "  send_arr_chars_arr_{} = None\n".format(i)

        if is_null_masked_type(typ):
            func_text += "  send_arr_nulls_{} = pre_shuffle_meta.send_arr_nulls_tup[{}]\n".format(i, i)
            func_text += "  if not is_contig:\n"
            func_text += "    n_bytes = (n_send + 7) >> 3\n"
            func_text += "    send_arr_nulls_{} = np.full(n_bytes + 2 * n_pes, 255, np.uint8)\n".format(i)
        else:
            func_text += "  send_arr_nulls_{} = None\n".format(i)

    send_buffs = ", ".join("send_buff_{}".format(i) for i in range(n_all))
    out_arrs = ", ".join("out_arr_{}".format(i) for i in range(n_all))
    all_comma = "," if n_all == 1 else ""
    send_counts_chars = ", ".join("send_counts_char_{}".format(i) for i in range(n_all))
    recv_counts_chars = ", ".join("recv_counts_char_{}".format(i) for i in range(n_all))
    send_arr_lens = ", ".join("send_arr_lens_{}".format(i) for i in range(n_all))
    send_arr_nulls = ", ".join("send_arr_nulls_{}".format(i) for i in range(n_all))
    send_arr_chars = ", ".join("send_arr_chars_{}".format(i) for i in range(n_all))
    send_disp_chars = ", ".join("send_disp_char_{}".format(i) for i in range(n_all))
    recv_disp_chars = ", ".join("recv_disp_char_{}".format(i) for i in range(n_all))
    tmp_offset_chars = ", ".join("tmp_offset_char_{}".format(i) for i in range(n_all))
    send_arr_chars_arrs = ", ".join("send_arr_chars_arr_{}".format(i) for i in range(n_all))


    func_text += ('  return ShuffleMeta(send_counts, recv_counts, n_send, '
        'n_out, send_disp, recv_disp, send_disp_nulls, recv_disp_nulls, '
        'tmp_offset, ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), ({}{}), '
        '({}{}), ({}{}), ({}{}), ({}{}), )\n').format(
            send_buffs, all_comma, out_arrs, all_comma, send_counts_chars,
            all_comma, recv_counts_chars, all_comma,
            send_arr_lens, all_comma, send_arr_nulls, all_comma,
            send_arr_chars, all_comma, send_disp_chars, all_comma,
            recv_disp_chars, all_comma,
            tmp_offset_chars, all_comma, send_arr_chars_arrs, all_comma
        )

    # print(func_text)

    loc_vars = {}
    exec(func_text, {'np': np, 'bodo': bodo,
         'pre_alloc_string_array': pre_alloc_string_array,
         'num_total_chars': num_total_chars,
         'get_data_ptr': get_data_ptr,
         'ShuffleMeta': ShuffleMeta,
         'get_ctypes_ptr': get_ctypes_ptr,
         'fix_cat_array_type':
         bodo.hiframes.pd_categorical_ext.fix_cat_array_type,
         'calc_disp_nulls': calc_disp_nulls}, loc_vars)
    finalize_impl = loc_vars['f']
    return finalize_impl


def alltoallv_tup(arrs, shuffle_meta, key_arrs):
    return arrs


@overload(alltoallv_tup)
def alltoallv_tup_overload(arrs, meta, key_arrs):
    n_keys = len(key_arrs.types)
    func_text = "def f(arrs, meta, key_arrs):\n"

    # null bitmap counts are send counts divided by 8
    if any(is_null_masked_type(t) for t in arrs.types):
        func_text += "  send_counts_nulls = np.empty(len(meta.send_counts), np.int32)\n"
        func_text += "  for i in range(len(meta.send_counts)):\n"
        func_text += "    send_counts_nulls[i] = (meta.send_counts[i] + 7) >> 3\n"
        func_text += "  recv_counts_nulls = np.empty(len(meta.recv_counts), np.int32)\n"
        func_text += "  for i in range(len(meta.recv_counts)):\n"
        func_text += "    recv_counts_nulls[i] = (meta.recv_counts[i] + 7) >> 3\n"
        func_text += "  tmp_null_bytes = np.empty(recv_counts_nulls.sum(), np.uint8)\n"

    for i, typ in enumerate(arrs.types):
        if isinstance(typ, (types.Array, IntegerArrayType, BooleanArrayType)):
            func_text += ("  bodo.libs.distributed_api.alltoallv("
                "meta.send_buff_tup[{}], meta.out_arr_tup[{}], meta.send_counts,"
                "meta.recv_counts, meta.send_disp, meta.recv_disp)\n").format(i, i)
        else:
            assert typ == string_array_type
            func_text += "  offset_ptr_{} = get_offset_ptr(meta.out_arr_tup[{}])\n".format(i, i)

            func_text += ("  bodo.libs.distributed_api.c_alltoallv("
                "meta.send_arr_lens_tup[{}].ctypes, offset_ptr_{}, meta.send_counts.ctypes, "
                "meta.recv_counts.ctypes, meta.send_disp.ctypes, "
                "meta.recv_disp.ctypes, int32_typ_enum)\n").format(i, i)

            func_text += ("  bodo.libs.distributed_api.c_alltoallv("
                "meta.send_arr_chars_tup[{}], get_data_ptr(meta.out_arr_tup[{}]), meta.send_counts_char_tup[{}].ctypes,"
                "meta.recv_counts_char_tup[{}].ctypes, meta.send_disp_char_tup[{}].ctypes,"
                "meta.recv_disp_char_tup[{}].ctypes, char_typ_enum)\n").format(i, i, i, i, i, i)

            func_text += "  convert_len_arr_to_offset(offset_ptr_{}, meta.n_out)\n".format(i)

        # XXX: handling null bits is only supported for data arrays now
        if i >= n_keys and is_null_masked_type(typ):
            func_text += "  null_bitmap_ptr_{} = get_arr_null_ptr(meta.out_arr_tup[{}])\n".format(i, i)
            func_text += ("  bodo.libs.distributed_api.c_alltoallv("
                "meta.send_arr_nulls_tup[{}].ctypes, tmp_null_bytes.ctypes, send_counts_nulls.ctypes, "
                "recv_counts_nulls.ctypes, meta.send_disp_nulls.ctypes, "
                "meta.recv_disp_nulls.ctypes, char_typ_enum)\n").format(i)

            func_text += "  copy_gathered_null_bytes(null_bitmap_ptr_{}, tmp_null_bytes, recv_counts_nulls, meta.recv_counts)\n".format(i)

    func_text += "  return ({}{})\n".format(
        ','.join(['meta.out_arr_tup[{}]'.format(i) for i in range(arrs.count)]),
        "," if arrs.count == 1 else "")
    # print(func_text)

    int32_typ_enum = np.int32(_numba_to_c_type_map[types.int32])
    char_typ_enum = np.int32(_numba_to_c_type_map[types.uint8])
    loc_vars = {}
    exec(func_text, {'np': np, 'bodo': bodo, 'get_offset_ptr': get_offset_ptr,
         'get_data_ptr': get_data_ptr, 'int32_typ_enum': int32_typ_enum,
         'char_typ_enum': char_typ_enum,
         'convert_len_arr_to_offset': convert_len_arr_to_offset,
         'copy_gathered_null_bytes':
            bodo.libs.distributed_api.copy_gathered_null_bytes,
         'get_arr_null_ptr': get_arr_null_ptr}, loc_vars)
    a2a_impl = loc_vars['f']
    return a2a_impl


def _get_keys_tup(recvs, key_arrs):
    return recvs[:len(key_arrs)]


@overload(_get_keys_tup)
def _get_keys_tup_overload(recvs, key_arrs):
    n_keys = len(key_arrs.types)
    func_text = "def f(recvs, key_arrs):\n"
    res = ",".join("recvs[{}]".format(i) for i in range(n_keys))
    func_text += "  return ({}{})\n".format(res, "," if n_keys==1 else "")
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars['f']
    return impl


def _get_data_tup(recvs, key_arrs):
    return recvs[len(key_arrs):]


@overload(_get_data_tup)
def _get_data_tup_overload(recvs, key_arrs):
    n_keys = len(key_arrs.types)
    n_all = len(recvs.types)
    n_data = n_all - n_keys
    func_text = "def f(recvs, key_arrs):\n"
    res = ",".join("recvs[{}]".format(i) for i in range(n_keys, n_all))
    func_text += "  return ({}{})\n".format(res, "," if n_data==1 else "")
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars['f']
    return impl


# returns scalar instead of tuple if only one array
def getitem_arr_tup_single(arrs, i):
    return arrs[0][i]


@overload(getitem_arr_tup_single)
def getitem_arr_tup_single_overload(arrs, i):
    if len(arrs.types) == 1:
        return lambda arrs, i: arrs[0][i]
    return lambda arrs, i: getitem_arr_tup(arrs, i)


def val_to_tup(val):
    return (val,)


@overload(val_to_tup)
def val_to_tup_overload(val):
    if isinstance(val, types.BaseTuple):
        return lambda val: val
    return lambda val: (val,)


def is_null_masked_type(t):
    return (t in (string_type, string_array_type)
        or isinstance(t, IntegerArrayType) or t == boolean_array)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_mask_bit(arr, i):
    if arr == string_array_type:
        return lambda arr, i: get_bit_bitmap(get_null_bitmap_ptr(arr), i)

    assert isinstance(arr, IntegerArrayType) or arr == boolean_array
    return lambda arr, i: bodo.libs.int_arr_ext.get_bit_bitmap_arr(
            arr._null_bitmap, i)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_arr_null_ptr(arr):
    if arr == string_array_type:
        return lambda arr: get_null_bitmap_ptr(arr)

    assert isinstance(arr, IntegerArrayType) or arr == boolean_array
    return lambda arr: arr._null_bitmap.ctypes
