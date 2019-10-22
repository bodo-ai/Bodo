# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
from numba import types, cgutils
from numba.targets.imputils import lower_builtin
from numba.targets.arrayobj import make_array
import bodo.io
from bodo.io import h5_api
from bodo.utils.utils import _numba_to_c_type_map
from bodo.io.h5_api import (
    h5file_type,
    h5dataset_or_group_type,
    h5dataset_type,
    h5group_type,
)
from bodo.libs.str_ext import string_type, gen_get_unicode_chars, gen_std_str_to_unicode

from llvmlite import ir as lir
import llvmlite.binding as ll
import bodo.io

if bodo.config._has_h5py:
    import h5py
    from bodo.io import _hdf5

    ll.add_symbol("h5_open", _hdf5.h5_open)
    ll.add_symbol(
        "h5_open_dset_or_group_obj", _hdf5.h5_open_dset_or_group_obj
    )
    ll.add_symbol("h5_read", _hdf5.h5_read)
    ll.add_symbol("h5_create_group", _hdf5.h5_create_group)
    ll.add_symbol("h5_write", _hdf5.h5_write)
    ll.add_symbol("h5_close", _hdf5.h5_close)
    ll.add_symbol("h5g_get_num_objs", _hdf5.h5g_get_num_objs)
    ll.add_symbol("h5g_get_objname_by_idx", _hdf5.h5g_get_objname_by_idx)
    ll.add_symbol("h5g_close", _hdf5.h5g_close)

h5file_lir_type = lir.IntType(64)

if bodo.config._has_h5py:
    # hid_t is 32bit in 1.8 but 64bit in 1.10
    if h5py.version.hdf5_version_tuple[1] == 8:
        h5file_lir_type = lir.IntType(32)
    else:
        assert h5py.version.hdf5_version_tuple[1] == 10

h5g_close = types.ExternalFunction("h5g_close", types.none(h5group_type))


@lower_builtin(operator.getitem, h5file_type, string_type)
@lower_builtin(operator.getitem, h5dataset_or_group_type, string_type)
def h5_open_dset_lower(context, builder, sig, args):
    fg_id, dset_name = args
    dset_name = gen_get_unicode_chars(context, builder, dset_name)

    fnty = lir.FunctionType(
        h5file_lir_type, [h5file_lir_type, lir.IntType(8).as_pointer()]
    )
    fn = builder.module.get_or_insert_function(
        fnty, name="h5_open_dset_or_group_obj"
    )
    return builder.call(fn, [fg_id, dset_name])


@lower_builtin(
    h5_api.h5write,
    h5dataset_type,
    types.int32,
    types.UniTuple,
    types.UniTuple,
    types.int64,
    types.Array,
)
def h5_write(context, builder, sig, args):
    # extra last arg type for type enum
    arg_typs = [
        h5file_lir_type,
        lir.IntType(32),
        lir.IntType(64).as_pointer(),
        lir.IntType(64).as_pointer(),
        lir.IntType(64),
        lir.IntType(8).as_pointer(),
        lir.IntType(32),
    ]
    fnty = lir.FunctionType(lir.IntType(32), arg_typs)

    fn = builder.module.get_or_insert_function(fnty, name="h5_write")
    out = make_array(sig.args[5])(context, builder, args[5])
    # store size vars array struct to pointer
    count_ptr = cgutils.alloca_once(builder, args[2].type)
    builder.store(args[2], count_ptr)
    size_ptr = cgutils.alloca_once(builder, args[3].type)
    builder.store(args[3], size_ptr)
    # store an int to specify data type
    typ_enum = _numba_to_c_type_map[sig.args[5].dtype]
    typ_arg = cgutils.alloca_once_value(
        builder, lir.Constant(lir.IntType(32), typ_enum)
    )
    call_args = [
        args[0],
        args[1],
        builder.bitcast(count_ptr, lir.IntType(64).as_pointer()),
        builder.bitcast(size_ptr, lir.IntType(64).as_pointer()),
        args[4],
        builder.bitcast(out.data, lir.IntType(8).as_pointer()),
        builder.load(typ_arg),
    ]

    return builder.call(fn, call_args)
