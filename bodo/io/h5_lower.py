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
