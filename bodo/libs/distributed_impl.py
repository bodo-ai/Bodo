"""Implementations for distributed operators. Loaded as needed to reduce import time."""

import numba
import numpy as np
from numba.core import types

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.pd_categorical_ext import CategoricalArrayType
from bodo.hiframes.time_ext import TimeArrayType
from bodo.libs.array import (
    array_info_type,
    array_to_info,
    cpp_table_to_py_table,
    delete_info,
    delete_table,
    info_to_array,
    py_data_to_cpp_table,
    table_type,
)
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
from bodo.mpi4py import MPI
from bodo.utils.typing import (
    BodoError,
    ColNamesMetaType,
    ExternalFunctionErrorChecked,
    MetaType,
    is_bodosql_context_type,
)
from bodo.utils.utils import (
    empty_like_type,
    is_array_typ,
    is_distributable_typ,
    numba_to_c_type,
)

DEFAULT_ROOT = 0


@numba.njit(cache=True)
def gatherv_impl_wrapper(
    data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
):
    return gatherv_impl_jit(data, allgather, warn_if_rep, root, comm)


# sendbuf, sendcount, recvbuf, recv_counts, displs, dtype
c_gatherv = types.ExternalFunction(
    "c_gatherv",
    types.void(
        types.voidptr,
        types.int64,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.int32,
        types.bool_,
        types.int32,
        types.int64,
    ),
)


_gather_table_py_entry = ExternalFunctionErrorChecked(
    "gather_table_py_entry",
    table_type(table_type, types.bool_, types.int32, types.int64),
)


_gather_array_py_entry = ExternalFunctionErrorChecked(
    "gather_array_py_entry",
    array_info_type(array_info_type, types.bool_, types.int32, types.int64),
)


@numba.generated_jit(nopython=True)
def gatherv_impl_jit(
    data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
):
    """gathers distributed data into rank 0 or all ranks if 'allgather' is set.
    'warn_if_rep' flag controls if a warning is raised if the input is replicated and
    gatherv has no effect (applicable only inside jit functions).
    """
    from bodo.libs.csr_matrix_ext import CSRMatrixType
    from bodo.libs.distributed_api import (
        Reduce_Type,
        bcast_scalar,
        bcast_tuple,
        gather_scalar,
    )

    bodo.hiframes.pd_dataframe_ext.check_runtime_cols_unsupported(
        data, "bodo.gatherv()"
    )

    if isinstance(data, bodo.hiframes.pd_series_ext.SeriesType):

        def impl(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            # get data and index arrays
            arr = bodo.hiframes.pd_series_ext.get_series_data(data)
            index = bodo.hiframes.pd_series_ext.get_series_index(data)
            name = bodo.hiframes.pd_series_ext.get_series_name(data)
            # Send name from workers to receiver in case of intercomm since not
            # available on receiver
            if comm != 0:
                bcast_root = MPI.PROC_NULL
                is_receiver = root == MPI.ROOT
                if is_receiver:
                    bcast_root = 0
                elif bodo.get_rank() == 0:
                    bcast_root = MPI.ROOT
                name = bcast_scalar(name, bcast_root, comm)
            # gather data
            out_arr = bodo.libs.distributed_api.gatherv(
                arr, allgather, warn_if_rep, root, comm
            )
            out_index = bodo.gatherv(index, allgather, warn_if_rep, root, comm)
            # create output Series
            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, name)

        return impl

    if isinstance(data, bodo.hiframes.pd_index_ext.RangeIndexType):
        INT64_MAX = np.iinfo(np.int64).max
        INT64_MIN = np.iinfo(np.int64).min

        def impl_range_index(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            is_receiver = bodo.get_rank() == root
            if comm != 0:
                is_receiver = root == MPI.ROOT

            # NOTE: assuming processes have chunks of a global RangeIndex with equal
            # steps. using min/max reductions to get start/stop of global range
            start = data._start
            stop = data._stop
            step = data._step
            name = data._name
            # Send name and step from workers to receiver in case of intercomm since not
            # available on receiver
            if comm != 0:
                bcast_root = MPI.PROC_NULL
                if is_receiver:
                    bcast_root = 0
                elif bodo.get_rank() == 0:
                    bcast_root = MPI.ROOT
                name = bcast_scalar(name, bcast_root, comm)
                step = bcast_scalar(step, bcast_root, comm)

            # ignore empty ranges coming from slicing, see test_getitem_slice
            if len(data) == 0:
                start = INT64_MAX
                stop = INT64_MIN
            min_op = np.int32(Reduce_Type.Min.value)
            max_op = np.int32(Reduce_Type.Max.value)
            start = bodo.libs.distributed_api.dist_reduce(
                start, min_op if step > 0 else max_op, comm
            )
            stop = bodo.libs.distributed_api.dist_reduce(
                stop, max_op if step > 0 else min_op, comm
            )
            total_len = bodo.libs.distributed_api.dist_reduce(
                len(data), np.int32(Reduce_Type.Sum.value), comm
            )
            # output is empty if all range chunks are empty
            if start == INT64_MAX and stop == INT64_MIN:
                start = 0
                stop = 0

            # make sure global length is consistent in case the user passes in incorrect
            # RangeIndex chunks (e.g. trivial index in each chunk), see test_rebalance
            l = max(0, -(-(stop - start) // step))
            if l < total_len:
                stop = start + step * total_len

            # gatherv() of dataframe returns 0-length arrays so index should
            # be 0-length to match
            if not is_receiver and not allgather:
                start = 0
                stop = 0

            return bodo.hiframes.pd_index_ext.init_range_index(start, stop, step, name)

        return impl_range_index

    # Index types
    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):
        from bodo.hiframes.pd_index_ext import PeriodIndexType

        if isinstance(data, PeriodIndexType):
            freq = data.freq

            def impl_pd_index(
                data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
            ):  # pragma: no cover
                arr = bodo.libs.distributed_api.gatherv(
                    data._data, allgather, warn_if_rep, root, comm
                )
                # Send name from workers to receiver in case of intercomm since not
                # available on receiver
                name = data._name
                if comm != 0:
                    bcast_root = MPI.PROC_NULL
                    is_receiver = root == MPI.ROOT
                    if is_receiver:
                        bcast_root = 0
                    elif bodo.get_rank() == 0:
                        bcast_root = MPI.ROOT
                    name = bcast_scalar(name, bcast_root, comm)
                return bodo.hiframes.pd_index_ext.init_period_index(arr, name, freq)

        else:

            def impl_pd_index(
                data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
            ):  # pragma: no cover
                arr = bodo.libs.distributed_api.gatherv(
                    data._data, allgather, warn_if_rep, root, comm
                )
                # Send name from workers to receiver in case of intercomm since not
                # available on receiver
                name = data._name
                if comm != 0:
                    bcast_root = MPI.PROC_NULL
                    is_receiver = root == MPI.ROOT
                    if is_receiver:
                        bcast_root = 0
                    elif bodo.get_rank() == 0:
                        bcast_root = MPI.ROOT
                    name = bcast_scalar(name, bcast_root, comm)
                return bodo.utils.conversion.index_from_array(arr, name)

        return impl_pd_index

    # MultiIndex index
    if isinstance(data, bodo.hiframes.pd_multi_index_ext.MultiIndexType):
        # just gather the data arrays
        # TODO: handle `levels` and `codes` when available
        def impl_multi_index(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            all_data = bodo.gatherv(data._data, allgather, warn_if_rep, root, comm)
            # Send name from workers to receiver in case of intercomm since not
            # available on receiver
            name = data._name
            names = data._names
            if comm != 0:
                bcast_root = MPI.PROC_NULL
                is_receiver = root == MPI.ROOT
                if is_receiver:
                    bcast_root = 0
                elif bodo.get_rank() == 0:
                    bcast_root = MPI.ROOT
                name = bcast_scalar(name, bcast_root, comm)
                names = bcast_tuple(names, bcast_root, comm)
            return bodo.hiframes.pd_multi_index_ext.init_multi_index(
                all_data, names, name
            )

        return impl_multi_index

    if isinstance(data, bodo.hiframes.table.TableType):
        table_type = data
        n_table_cols = len(table_type.arr_types)
        in_col_inds = MetaType(tuple(range(n_table_cols)))
        out_cols_arr = np.array(range(n_table_cols), dtype=np.int64)

        def impl(data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0):
            cpp_table = py_data_to_cpp_table(data, (), in_col_inds, n_table_cols)
            out_cpp_table = _gather_table_py_entry(cpp_table, allgather, root, comm)
            ret = cpp_table_to_py_table(out_cpp_table, out_cols_arr, table_type, 0)
            delete_table(out_cpp_table)
            return ret

        return impl

    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        n_cols = len(data.columns)
        # empty dataframe case
        if n_cols == 0:
            __col_name_meta_value_gatherv_no_cols = ColNamesMetaType(())

            def impl(
                data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
            ):  # pragma: no cover
                index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)
                g_index = bodo.gatherv(index, allgather, warn_if_rep, root, comm)
                return bodo.hiframes.pd_dataframe_ext.init_dataframe(
                    (), g_index, __col_name_meta_value_gatherv_no_cols
                )

            return impl

        data_args = ", ".join(f"g_data_{i}" for i in range(n_cols))

        func_text = f"def impl_df(data, allgather=False, warn_if_rep=True, root={DEFAULT_ROOT}, comm=0):\n"
        if data.is_table_format:
            data_args = "T2"
            func_text += (
                "  T = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(data)\n"
                "  T2 = bodo.gatherv(T, allgather, warn_if_rep, root, comm)\n"
            )
        else:
            for i in range(n_cols):
                func_text += f"  data_{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(data, {i})\n"
                func_text += f"  g_data_{i} = bodo.gatherv(data_{i}, allgather, warn_if_rep, root, comm)\n"
        func_text += (
            "  index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)\n"
            "  g_index = bodo.gatherv(index, allgather, warn_if_rep, root, comm)\n"
            f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_args},), g_index, __col_name_meta_value_gatherv_with_cols)\n"
        )

        loc_vars = {}
        glbls = {
            "bodo": bodo,
            "__col_name_meta_value_gatherv_with_cols": ColNamesMetaType(data.columns),
        }
        exec(func_text, glbls, loc_vars)
        impl_df = loc_vars["impl_df"]
        return impl_df

    # CSR Matrix
    if isinstance(data, CSRMatrixType):

        def impl_csr_matrix(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            # gather local data
            all_data = bodo.gatherv(data.data, allgather, warn_if_rep, root, comm)
            all_col_inds = bodo.gatherv(
                data.indices, allgather, warn_if_rep, root, comm
            )
            all_indptr = bodo.gatherv(data.indptr, allgather, warn_if_rep, root, comm)
            all_local_rows = gather_scalar(
                data.shape[0], allgather, root=root, comm=comm
            )
            n_rows = all_local_rows.sum()
            n_cols = bodo.libs.distributed_api.dist_reduce(
                data.shape[1], np.int32(Reduce_Type.Max.value), comm
            )

            # using np.int64 in output since maximum index value is not known at
            # compilation time
            new_indptr = np.empty(n_rows + 1, np.int64)
            all_col_inds = all_col_inds.astype(np.int64)

            # construct indptr for output
            new_indptr[0] = 0
            out_ind = 1  # current position in output new_indptr
            indptr_ind = 0  # current position in input all_indptr
            for n_loc_rows in all_local_rows:
                for _ in range(n_loc_rows):
                    row_size = all_indptr[indptr_ind + 1] - all_indptr[indptr_ind]
                    new_indptr[out_ind] = new_indptr[out_ind - 1] + row_size
                    out_ind += 1
                    indptr_ind += 1
                indptr_ind += 1  # skip extra since each arr is n_rows + 1

            return bodo.libs.csr_matrix_ext.init_csr_matrix(
                all_data, all_col_inds, new_indptr, (n_rows, n_cols)
            )

        return impl_csr_matrix

    # Tuple of data containers
    if isinstance(data, types.BaseTuple):
        func_text = f"def impl_tuple(data, allgather=False, warn_if_rep=True, root={DEFAULT_ROOT}, comm=0):\n"
        func_text += "  return ({}{})\n".format(
            ", ".join(
                f"bodo.gatherv(data[{i}], allgather, warn_if_rep, root, comm)"
                for i in range(len(data))
            ),
            "," if len(data) > 0 else "",
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl_tuple = loc_vars["impl_tuple"]
        return impl_tuple

    if data is types.none:
        return (
            lambda data,
            allgather=False,
            warn_if_rep=True,
            root=DEFAULT_ROOT,
            comm=0: None
        )  # pragma: no cover

    if isinstance(data, types.Array) and data.ndim != 1:
        typ_val = numba_to_c_type(data.dtype)

        def gatherv_impl(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            data = np.ascontiguousarray(data)
            rank = bodo.get_rank()
            is_receiver = rank == root
            is_intercomm = comm != 0
            if is_intercomm:
                is_receiver = root == MPI.ROOT
            # size to handle multi-dim arrays
            n_loc = data.size
            recv_counts = gather_scalar(
                np.int64(n_loc), allgather, root=root, comm=comm
            )
            n_total = recv_counts.sum()
            all_data = empty_like_type(n_total, data)
            # displacements
            displs = np.empty(1, np.int64)
            if is_receiver or allgather:
                displs = bodo.ir.join.calc_disp(recv_counts)
            c_gatherv(
                data.ctypes,
                np.int64(n_loc),
                all_data.ctypes,
                recv_counts.ctypes,
                displs.ctypes,
                np.int32(typ_val),
                allgather,
                np.int32(root),
                comm,
            )

            shape = data.shape
            # Send shape from workers to receiver in case of intercomm since not
            # available on receiver
            if is_intercomm:
                bcast_root = MPI.PROC_NULL
                if is_receiver:
                    bcast_root = 0
                elif rank == 0:
                    bcast_root = MPI.ROOT
                shape = bcast_tuple(shape, bcast_root, comm)

            # handle multi-dim case
            return all_data.reshape((-1,) + shape[1:])

        return gatherv_impl

    if isinstance(data, CategoricalArrayType):

        def impl_cat(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            codes = bodo.gatherv(data.codes, allgather, warn_if_rep, root, comm)
            return bodo.hiframes.pd_categorical_ext.init_categorical_array(
                codes, data.dtype
            )

        return impl_cat

    if isinstance(data, bodo.MatrixType):

        def impl_matrix(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            new_data = bodo.gatherv(data.data, allgather, warn_if_rep, root, comm)
            return bodo.libs.matrix_ext.init_np_matrix(new_data)

        return impl_matrix

    if is_array_typ(data, False):
        dtype = data

        def impl(data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0):
            input_info = array_to_info(data)
            out_info = _gather_array_py_entry(input_info, allgather, root, comm)
            ret = info_to_array(out_info, dtype)
            delete_info(out_info)
            return ret

        return impl

    # List of distributable data
    if isinstance(data, types.List) and is_distributable_typ(data.dtype):

        def impl_list(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank = bodo.get_rank()
            is_receiver = rank == root
            is_intercomm = comm != 0
            if is_intercomm:
                is_receiver = root == MPI.ROOT

            length = len(data)
            # Send length from workers to receiver in case of intercomm since not
            # available on receiver
            if is_intercomm:
                bcast_root = MPI.PROC_NULL
                if is_receiver:
                    bcast_root = 0
                elif rank == 0:
                    bcast_root = MPI.ROOT
                length = bcast_scalar(length, bcast_root, comm)

            out = []
            for i in range(length):
                in_val = data[i] if not is_receiver else data[0]
                out.append(bodo.gatherv(in_val, allgather, warn_if_rep, root, comm))

            return out

        return impl_list

    # Dict of distributable data
    if isinstance(data, types.DictType) and is_distributable_typ(data.value_type):

        def impl_dict(
            data, allgather=False, warn_if_rep=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank = bodo.get_rank()
            is_receiver = rank == root
            is_intercomm = comm != 0
            if is_intercomm:
                is_receiver = root == MPI.ROOT

            length = len(data)
            # Send length from workers to receiver in case of intercomm since not
            # available on receiver
            if is_intercomm:
                bcast_root = MPI.PROC_NULL
                if is_receiver:
                    bcast_root = 0
                elif rank == 0:
                    bcast_root = MPI.ROOT
                length = bcast_scalar(length, bcast_root, comm)

            in_keys = list(data.keys())
            in_values = list(data.values())
            out = {}
            for i in range(length):
                key = in_keys[i] if not is_receiver else in_keys[0]
                if is_intercomm:
                    bcast_root = MPI.PROC_NULL
                    if is_receiver:
                        bcast_root = 0
                    elif rank == 0:
                        bcast_root = MPI.ROOT
                    key = bcast_scalar(key, bcast_root, comm)
                value = in_values[i] if not is_receiver else in_values[0]
                out[key] = bodo.gatherv(value, allgather, warn_if_rep, root, comm)

            return out

        return impl_dict

    if is_bodosql_context_type(data):
        import bodosql

        func_text = f"def impl_bodosql_context(data, allgather=False, warn_if_rep=True, root={DEFAULT_ROOT}, comm=0):\n"
        comma_sep_names = ", ".join([f"'{name}'" for name in data.names])
        comma_sep_dfs = ", ".join(
            [
                f"bodo.gatherv(data.dataframes[{i}], allgather, warn_if_rep, root, comm)"
                for i in range(len(data.dataframes))
            ]
        )
        func_text += f"  return bodosql.context_ext.init_sql_context(({comma_sep_names}, ), ({comma_sep_dfs}, ), data.catalog, None)\n"
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)
        impl_bodosql_context = loc_vars["impl_bodosql_context"]
        return impl_bodosql_context

    if type(data).__name__ == "TablePathType":
        try:
            from bodosql import TablePathType
        except ImportError:  # pragma: no cover
            raise ImportError("Install bodosql to use gatherv() with TablePathType")
        assert isinstance(data, TablePathType)
        # Table Path info is all compile time so we return the same data.
        func_text = f"def impl_table_path(data, allgather=False, warn_if_rep=True, root={DEFAULT_ROOT}, comm=0):\n"
        func_text += "  return data\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        impl_table_path = loc_vars["impl_table_path"]
        return impl_table_path

    raise BodoError(f"gatherv() not available for {data}")  # pragma: no cover


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
