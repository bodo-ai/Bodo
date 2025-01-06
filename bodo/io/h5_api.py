import operator

import numba
import numpy as np
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.typing import signature
from numba.core.typing.templates import AbstractTemplate, infer_global
from numba.extending import (
    intrinsic,
    models,
    overload,
    overload_method,
    register_model,
)

import bodo
import bodo.io
from bodo.libs.str_ext import (
    std_str_to_unicode,
    std_str_type,
    string_type,
    unicode_to_utf8,
)
from bodo.utils.typing import is_overload_none, parse_dtype
from bodo.utils.utils import numba_to_c_type

if bodo.utils.utils.has_supported_h5py():
    import h5py
    import llvmlite.binding as ll

    from bodo.ext import _hdf5

    ll.add_symbol("h5_open", _hdf5.h5_open)
    ll.add_symbol("h5_open_dset_or_group_obj", _hdf5.h5_open_dset_or_group_obj)
    ll.add_symbol("h5_read", _hdf5.h5_read)
    ll.add_symbol("h5_create_group", _hdf5.h5_create_group)
    ll.add_symbol("h5_write", _hdf5.h5_write)
    ll.add_symbol("h5_close", _hdf5.h5_close)
    ll.add_symbol("h5g_get_num_objs", _hdf5.h5g_get_num_objs)
    ll.add_symbol("h5g_get_objname_by_idx", _hdf5.h5g_get_objname_by_idx)
    ll.add_symbol("h5g_close", _hdf5.h5g_close)
    ll.add_symbol("h5_read_filter", _hdf5.h5_read_filter)
    ll.add_symbol("h5_size", _hdf5.h5_size)
    ll.add_symbol("h5_create_dset", _hdf5.h5_create_dset)


################## Types #######################


class H5FileType(types.Opaque):
    def __init__(self):
        super().__init__(name="H5FileType")


h5file_type = H5FileType()


class H5DatasetType(types.Opaque):
    def __init__(self):
        super().__init__(name="H5DatasetType")


h5dataset_type = H5DatasetType()


class H5GroupType(types.Opaque):
    def __init__(self):
        super().__init__(name="H5GroupType")


h5group_type = H5GroupType()


class H5DatasetOrGroupType(types.Opaque):
    def __init__(self):
        super().__init__(name="H5DatasetOrGroupType")


h5dataset_or_group_type = H5DatasetOrGroupType()

h5file_data_type = types.int64


@register_model(H5FileType)
@register_model(H5DatasetType)
@register_model(H5GroupType)
@register_model(H5DatasetOrGroupType)
class H5FileModel(models.IntegerModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, h5file_data_type)


# type for list of names
string_list_type = types.List(string_type)


#################################################


@intrinsic
def unify_h5_id(typingctx, tp):
    """converts h5 id objects (which all have the same hid_t representation) to a single
    type to enable reuse of external functions.
    """

    def codegen(context, builder, sig, args):
        return args[0]

    return h5file_type(tp), codegen


@intrinsic
def cast_to_h5_dset(typingctx, tp):
    """converts h5dataset_or_group_type to h5dataset_type"""
    assert tp in (h5dataset_type, h5dataset_or_group_type)

    def codegen(context, builder, sig, args):
        return args[0]

    return h5dataset_type(tp), codegen


h5_open = types.ExternalFunction(
    "h5_open", h5file_type(types.voidptr, types.voidptr, types.int64)
)


if bodo.utils.utils.has_supported_h5py():

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
        _is_parallel=0,  # bodo internal flag
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
            _is_parallel=0,  # bodo internal flag
        ):  # pragma: no cover
            if mode is None:
                mode = "a"  # TODO: support and test
            f = h5_open(unicode_to_utf8(name), unicode_to_utf8(mode), _is_parallel)
            return f

        return impl


h5_close = types.ExternalFunction("h5_close", types.int32(h5file_type))


@overload_method(H5FileType, "close", no_unliteral=True)
def overload_h5_file(f):
    def impl(f):
        h5_close(f)

    return impl


@overload_method(H5FileType, "keys", no_unliteral=True)
@overload_method(H5DatasetOrGroupType, "keys", no_unliteral=True)
def overload_h5_file_keys(obj_id):
    def h5f_keys_impl(obj_id):  # pragma: no cover
        obj_name_list = []
        nobjs = h5g_get_num_objs(obj_id)
        for i in range(nobjs):
            obj_name = h5g_get_objname_by_idx(obj_id, i)
            obj_name_list.append(obj_name)
        return obj_name_list

    return h5f_keys_impl


h5_create_dset = types.ExternalFunction(
    "h5_create_dset",
    h5dataset_type(h5file_type, types.voidptr, types.int32, types.voidptr, types.int32),
)


@overload_method(H5FileType, "create_dataset", no_unliteral=True)
@overload_method(H5GroupType, "create_dataset", no_unliteral=True)
def overload_h5_file_create_dataset(obj_id, name, shape=None, dtype=None, data=None):
    assert is_overload_none(data)  # TODO: support passing data directly
    # TODO: support non-constant dtype string value
    nb_dtype = parse_dtype(dtype)
    typ_enum = np.int32(numba_to_c_type(nb_dtype))
    ndim = np.int32(len(shape))

    def impl(obj_id, name, shape=None, dtype=None, data=None):  # pragma: no cover
        counts = np.asarray(shape)
        return h5_create_dset(
            unify_h5_id(obj_id), unicode_to_utf8(name), ndim, counts.ctypes, typ_enum
        )

    return impl


h5_create_group = types.ExternalFunction(
    "h5_create_group", h5group_type(h5file_type, types.voidptr)
)


@overload_method(H5FileType, "create_group", no_unliteral=True)
@overload_method(H5GroupType, "create_group", no_unliteral=True)
def overload_h5_file_create_group(obj_id, name, track_order=None):
    assert is_overload_none(track_order)  # TODO: support?

    def impl(obj_id, name, track_order=None):  # pragma: no cover
        return h5_create_group(unify_h5_id(obj_id), unicode_to_utf8(name))

    return impl


h5_open_dset_or_group_obj = types.ExternalFunction(
    "h5_open_dset_or_group_obj", h5dataset_or_group_type(h5file_type, types.voidptr)
)


@overload(operator.getitem)
def overload_getitem_file(in_f, in_idx):
    if in_f in (h5file_type, h5dataset_or_group_type) and in_idx == string_type:

        def impl(in_f, in_idx):  # pragma: no cover
            d = h5_open_dset_or_group_obj(unify_h5_id(in_f), unicode_to_utf8(in_idx))
            return d

        return impl


@infer_global(operator.setitem)
class SetItemH5Dset(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        if args[0] == h5dataset_type:
            return signature(types.none, *args)


h5g_close = types.ExternalFunction("h5g_close", types.none(h5group_type))
_h5g_get_num_objs = types.ExternalFunction("h5g_get_num_objs", types.int64(h5file_type))


@numba.njit
def h5g_get_num_objs(obj_id):  # pragma: no cover
    return _h5g_get_num_objs(unify_h5_id(obj_id))


_h5g_get_objname_by_idx = types.ExternalFunction(
    "h5g_get_objname_by_idx", std_str_type(h5file_type, types.int64)
)


@numba.njit
def h5g_get_objname_by_idx(obj_id, ind):  # pragma: no cover
    return std_str_to_unicode(_h5g_get_objname_by_idx(unify_h5_id(obj_id), ind))


_h5_read = types.ExternalFunction(
    "h5_read",
    types.int32(
        h5dataset_type,
        types.int32,
        types.voidptr,
        types.voidptr,
        types.int64,
        types.voidptr,
        types.int32,
    ),
)


@numba.njit
def h5read(dset_id, ndim, starts, counts, is_parallel, out_arr):  # pragma: no cover
    starts_ptr = tuple_to_ptr(starts)
    counts_ptr = tuple_to_ptr(counts)
    type_enum = bodo.libs.distributed_api.get_type_enum(out_arr)
    return _h5_read(
        cast_to_h5_dset(dset_id),
        ndim,
        starts_ptr,
        counts_ptr,
        is_parallel,
        out_arr.ctypes,
        type_enum,
    )


_h5_write = types.ExternalFunction(
    "h5_write",
    types.int32(
        h5dataset_type,
        types.int32,
        types.voidptr,
        types.voidptr,
        types.int64,
        types.voidptr,
        types.int32,
    ),
)


@numba.njit
def h5write(dset_id, ndim, starts, counts, is_parallel, out_arr):  # pragma: no cover
    starts_ptr = tuple_to_ptr(starts)
    counts_ptr = tuple_to_ptr(counts)
    type_enum = bodo.libs.distributed_api.get_type_enum(out_arr)
    return _h5_write(
        cast_to_h5_dset(dset_id),
        ndim,
        starts_ptr,
        counts_ptr,
        is_parallel,
        out_arr.ctypes,
        type_enum,
    )


def h5_read_dummy():
    return


@infer_global(h5_read_dummy)
class H5ReadType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        ndim = args[1].literal_value
        dtype = getattr(types, args[2].literal_value)
        ret_typ = types.Array(dtype, ndim, "C")
        index = args[3]
        # the output type may not be the same as the whole dataset due to slice
        # selection, e.g. A[:,2,:]
        ret_typ = self.context.resolve_function_type(
            operator.getitem, [ret_typ, index], {}
        ).return_type.copy(layout="C")
        return signature(ret_typ, *args)


H5ReadType._no_unliteral = True


_h5size = types.ExternalFunction("h5_size", types.int64(h5dataset_type, types.int32))


@numba.njit
def h5size(obj_id, dim_ind):  # pragma: no cover
    return _h5size(cast_to_h5_dset(obj_id), dim_ind)


sum_op = bodo.libs.distributed_api.Reduce_Type.Sum.value


@numba.njit
def get_filter_read_indices(bool_arr):  # pragma: no cover
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
    bodo.libs.distributed_api.bcast_preallocated(all_indices)

    start = bodo.libs.distributed_api.get_start(n, n_pes, rank)
    end = bodo.libs.distributed_api.get_end(n, n_pes, rank)
    return all_indices[start:end]


@intrinsic(prefer_literal=True)
def tuple_to_ptr(typingctx, tuple_tp):
    """
    Converts a tuple representation to a pointer.
    Note: The tuple may contain Integer literals, which changes
    its value, so prefer_literal=True is required.
    """

    def codegen(context, builder, sig, args):
        ptr = cgutils.alloca_once(builder, args[0].type)
        builder.store(args[0], ptr)
        return builder.bitcast(ptr, lir.IntType(8).as_pointer())

    return signature(types.voidptr, tuple_tp), codegen


_h5read_filter = types.ExternalFunction(
    "h5_read_filter",
    types.int32(
        h5dataset_type,
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
def h5read_filter(
    dset_id, ndim, starts, counts, is_parallel, out_arr, read_indices
):  # pragma: no cover
    starts_ptr = tuple_to_ptr(starts)
    counts_ptr = tuple_to_ptr(counts)
    type_enum = bodo.libs.distributed_api.get_type_enum(out_arr)
    return _h5read_filter(
        cast_to_h5_dset(dset_id),
        ndim,
        starts_ptr,
        counts_ptr,
        is_parallel,
        out_arr.ctypes,
        type_enum,
        read_indices.ctypes,
        len(read_indices),
    )
