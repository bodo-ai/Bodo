# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
import numpy as np
import numba
from numba import types, cgutils
from numba.typing.templates import (
    infer_global,
    AbstractTemplate,
    AttributeTemplate,
    bound_function,
)
from numba.typing import signature
from llvmlite import ir as lir
from numba.extending import (
    register_model,
    models,
    infer_getattr,
    infer,
    intrinsic,
    overload,
    overload_method,
)
from bodo.libs.str_ext import string_type, string_to_char_ptr
import bodo
from bodo.utils.utils import unliteral_all, _numba_to_c_type_map
from bodo.utils.typing import parse_dtype, is_overload_none
import bodo.io

if bodo.config._has_h5py:
    import h5py
    from bodo.io import _hdf5
    import llvmlite.binding as ll

    ll.add_symbol("h5_read_filter", _hdf5.h5_read_filter)
    ll.add_symbol("h5_size", _hdf5.h5_size)
    ll.add_symbol("h5_create_dset", _hdf5.h5_create_dset)


################## Types #######################


class H5FileType(types.Opaque):
    def __init__(self):
        super(H5FileType, self).__init__(name="H5FileType")


h5file_type = H5FileType()


class H5DatasetType(types.Opaque):
    def __init__(self):
        super(H5DatasetType, self).__init__(name="H5DatasetType")


h5dataset_type = H5DatasetType()


class H5GroupType(types.Opaque):
    def __init__(self):
        super(H5GroupType, self).__init__(name="H5GroupType")


h5group_type = H5GroupType()


class H5DatasetOrGroupType(types.Opaque):
    def __init__(self):
        super(H5DatasetOrGroupType, self).__init__(name="H5DatasetOrGroupType")


h5dataset_or_group_type = H5DatasetOrGroupType()

h5file_data_type = types.int64

if bodo.config._has_h5py:
    # hid_t is 32bit in 1.8 but 64bit in 1.10
    if h5py.version.hdf5_version_tuple[1] == 8:
        h5file_data_type = types.int32
    else:
        assert h5py.version.hdf5_version_tuple[1] == 10


@register_model(H5FileType)
@register_model(H5DatasetType)
@register_model(H5GroupType)
@register_model(H5DatasetOrGroupType)
class H5FileModel(models.IntegerModel):
    def __init__(self, dmm, fe_type):
        super(H5FileModel, self).__init__(dmm, h5file_data_type)


# type for list of names
string_list_type = types.List(string_type)


#################################################



@intrinsic
def unify_h5_id(typingctx, tp=None):
    """converts h5 id objects (which all have the same hid_t representation) to a single
    type to enable reuse of external functions.
    """
    def codegen(context, builder, sig, args):
        return args[0]
    return h5file_type(tp), codegen



h5_open = types.ExternalFunction("h5_open", h5file_type(types.voidptr, types.voidptr))

if bodo.config._has_h5py:

    @overload(h5py.File)
    def overload_h5py_file(
        name,
        mode=None,
        driver=None,
        libver=None,
        userblock_size=None,
        swmr=False,
        rdcc_nslots=None,
        rdcc_nbytes=None,
        rdcc_w0=None,
        track_order=None,
    ):
        def impl(
            name,
            mode=None,
            driver=None,
            libver=None,
            userblock_size=None,
            swmr=False,
            rdcc_nslots=None,
            rdcc_nbytes=None,
            rdcc_w0=None,
            track_order=None,
        ):
            if mode is None:
                mode = "a"
            return h5_open(name._data, mode._data)

        return impl


h5_close = types.ExternalFunction("h5_close", types.int32(h5file_type))


@overload_method(H5FileType, "close")
def overload_h5_file(f):
    def impl(f):
        h5_close(f)
    return impl


@overload_method(H5FileType, "keys")
@overload_method(H5DatasetOrGroupType, "keys")
def overload_h5_file_keys(obj_id):
    def h5f_keys_impl(obj_id):
        obj_name_list = []
        nobjs = h5g_get_num_objs(obj_id)
        for i in range(nobjs):
            obj_name = h5g_get_objname_by_idx(obj_id, i)
            obj_name_list.append(obj_name)
        return obj_name_list
    return h5f_keys_impl


h5_create_dset = types.ExternalFunction("h5_create_dset", h5dataset_type(h5file_type,
    types.voidptr, types.int32, types.voidptr, types.int32))


@overload_method(H5FileType, "create_dataset")
@overload_method(H5GroupType, "create_dataset")
def overload_h5_file_create_dataset(obj_id, name, shape=None, dtype=None, data=None):
    assert is_overload_none(data)  # TODO: support passing data directly
    # TODO: support non-constant dtype string value
    nb_dtype = parse_dtype(dtype)
    typ_enum = np.int32(_numba_to_c_type_map[nb_dtype])
    ndim = np.int32(len(shape))

    def impl(obj_id, name, shape=None, dtype=None, data=None):
        counts = np.asarray(shape)
        return h5_create_dset(
            unify_h5_id(obj_id), string_to_char_ptr(name), ndim, counts.ctypes, typ_enum)
    return impl


h5_create_group = types.ExternalFunction("h5_create_group", h5group_type(h5file_type,
    types.voidptr))


@overload_method(H5FileType, "create_group")
@overload_method(H5GroupType, "create_group")
def overload_h5_file_create_group(obj_id, name, track_order=None):
    assert is_overload_none(track_order)  # TODO: support?

    def impl(obj_id, name, track_order=None):
        return h5_create_group(unify_h5_id(obj_id), string_to_char_ptr(name))
    return impl


@infer_global(operator.getitem)
class GetItemH5File(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        assert not kws
        (in_f, in_idx) = args
        if in_f == h5file_type:
            assert in_idx == string_type
            return signature(h5dataset_or_group_type, in_f, in_idx)
        if in_f == h5dataset_or_group_type and in_idx == string_type:
            return signature(h5dataset_or_group_type, in_f, in_idx)


@infer_global(operator.setitem)
class SetItemH5Dset(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        if args[0] == h5dataset_type:
            return signature(types.none, *args)



_h5g_get_num_objs = types.ExternalFunction("h5g_get_num_objs", types.int64(h5file_type))


@numba.njit
def h5g_get_num_objs(obj_id):
    return _h5g_get_num_objs(unify_h5_id(obj_id))


def h5g_get_objname_by_idx():
    return


def h5read():
    """dummy function for C h5_read"""
    return


def h5create_dset():
    """dummy function for C h5_create_dset"""
    return


def h5create_group():
    """dummy function for C h5create_group"""
    return


def h5write():
    """dummy function for C h5_write"""
    return


def h5_read_dummy():
    return


@infer_global(h5_read_dummy)
class H5ReadType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ndim = args[1].literal_value
        dtype = getattr(types, args[2].literal_value)
        ret_typ = types.Array(dtype, ndim, "C")
        return signature(ret_typ, *args)


h5size = types.ExternalFunction(
    "h5_size", types.int64(h5dataset_or_group_type, types.int32)
)


@infer_global(h5read)
class H5Read(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 6
        return signature(types.int32, *unliteral_all(args))


@infer_global(h5create_dset)
class H5CreateDSet(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 4
        return signature(h5file_type, *unliteral_all(args))


@infer_global(h5create_group)
class H5CreateGroup(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(h5file_type, *unliteral_all(args))


@infer_global(h5write)
class H5Write(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 6
        return signature(types.int32, *unliteral_all(args))


@infer_global(h5g_get_objname_by_idx)
class H5GgetObjNameByIdx(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        return signature(string_type, *args)


sum_op = bodo.libs.distributed_api.Reduce_Type.Sum.value


@numba.njit
def get_filter_read_indices(bool_arr):
    indices = bool_arr.nonzero()[0]
    rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()

    # get number of elements before this processor to align the indices
    # assuming bool_arr can be 1D_Var
    all_starts = np.empty(n_pes, np.int64)
    n_bool = len(bool_arr)
    bodo.libs.distributed_api.allgather(all_starts, n_bool)
    ind_start = all_starts.cumsum()[rank] - n_bool
    # n_arr = bodo.libs.distributed_api.dist_reduce(len(bool_arr), np.int32(sum_op))
    # ind_start = bodo.libs.distributed_api.get_start(n_arr, n_pes, rank)
    indices += ind_start

    # TODO: use prefix-sum and all-to-all
    # all_indices = np.empty(n, indices.dtype)
    # allgatherv(all_indices, indices)
    n = bodo.libs.distributed_api.dist_reduce(len(indices), np.int32(sum_op))
    inds = bodo.libs.distributed_api.gatherv(indices)
    if rank == 0:
        all_indices = inds
    else:
        all_indices = np.empty(n, indices.dtype)
    bodo.libs.distributed_api.bcast(all_indices)

    start = bodo.libs.distributed_api.get_start(n, n_pes, rank)
    end = bodo.libs.distributed_api.get_end(n, n_pes, rank)
    return all_indices[start:end]


@intrinsic
def tuple_to_ptr(typingctx, tuple_tp=None):
    def codegen(context, builder, sig, args):
        ptr = cgutils.alloca_once(builder, args[0].type)
        builder.store(args[0], ptr)
        return builder.bitcast(ptr, lir.IntType(8).as_pointer())

    return signature(types.voidptr, tuple_tp), codegen


_h5read_filter = types.ExternalFunction(
    "h5_read_filter",
    types.int32(
        h5dataset_or_group_type,
        types.int32,
        types.voidptr,
        types.voidptr,
        types.intp,
        types.voidptr,
        types.int32,
        types.voidptr,
        types.int32,
    ),
)


@numba.njit
def h5read_filter(dset_id, ndim, starts, counts, is_parallel, out_arr, read_indices):
    starts_ptr = tuple_to_ptr(starts)
    counts_ptr = tuple_to_ptr(counts)
    type_enum = bodo.libs.distributed_api.get_type_enum(out_arr)
    return _h5read_filter(
        dset_id,
        ndim,
        starts_ptr,
        counts_ptr,
        is_parallel,
        out_arr.ctypes,
        type_enum,
        read_indices.ctypes,
        len(read_indices),
    )
