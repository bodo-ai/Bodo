"""Implementations for distributed operators. Loaded as needed to reduce import time."""

import numba
import numpy as np
from mpi4py import MPI
from numba.core import types

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.datetime_timedelta_ext import timedelta_array_type
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
    ArrayItemArrayType,
    np_offset_type,
    offset_type,
)
from bodo.libs.binary_arr_ext import binary_array_type
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType, set_bit_to_arr
from bodo.libs.interval_arr_ext import IntervalArrayType
from bodo.libs.pd_datetime_arr_ext import DatetimeArrayType
from bodo.libs.str_arr_ext import (
    convert_len_arr_to_offset,
    get_bit_bitmap,
    string_array_type,
)
from bodo.utils.typing import (
    BodoError,
    ColNamesMetaType,
    ExternalFunctionErrorChecked,
    MetaType,
    is_bodosql_context_type,
    is_overload_none,
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


# sendbuff, sendcounts, displs, recvbuf, recv_count, dtype
c_scatterv = types.ExternalFunction(
    "c_scatterv",
    types.void(
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.int64,
        types.int32,
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

    if isinstance(data, bodo.types.MatrixType):

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
            import bodosql.compiler  # isort:skip # noqa
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


@numba.njit(cache=True)
def scatterv_impl(data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0):
    return scatterv_impl_jit(data, send_counts, warn_if_dist, root, comm)


@numba.generated_jit(nopython=True)
def scatterv_impl_jit(
    data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
):
    """nopython implementation of scatterv()"""
    from bodo.libs.distributed_api import bcast_scalar, bcast_tuple

    if isinstance(data, types.Array):
        return (
            lambda data,
            send_counts=None,
            warn_if_dist=True,
            root=DEFAULT_ROOT,
            comm=0: _scatterv_np(data, send_counts, warn_if_dist, root, comm)
        )  # pragma: no cover

    if data in (string_array_type, binary_array_type):
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))
        empty_int32_arr = np.array([], np.int32)

        if data == binary_array_type:
            alloc_fn = bodo.libs.binary_arr_ext.pre_alloc_binary_array
        else:
            alloc_fn = bodo.libs.str_arr_ext.pre_alloc_string_array

        def impl(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank, is_intercomm, is_sender, n_pes = get_scatter_comm_info(root, comm)

            n_all = bodo.libs.distributed_api.bcast_scalar(len(data), root, comm)

            # convert offsets to lengths of strings
            send_arr_lens = np.empty(
                len(data), np.uint32
            )  # XXX offset type is offset_type, lengths for comm are uint32
            for i in range(len(data)):
                send_arr_lens[i] = bodo.libs.str_arr_ext.get_str_arr_item_length(
                    data, i
                )

            # ------- calculate buffer counts -------

            send_counts = bodo.libs.distributed_impl._get_scatterv_send_counts(
                send_counts, n_pes, n_all
            )

            # displacements
            displs = bodo.ir.join.calc_disp(send_counts)

            # compute send counts for characters
            send_counts_char = np.empty(n_pes, np.int64)
            if is_sender:
                curr_str = 0
                for i in range(n_pes):
                    c = 0
                    for _ in range(send_counts[i]):
                        c += send_arr_lens[curr_str]
                        curr_str += 1
                    send_counts_char[i] = c

            send_counts_char = bodo.libs.distributed_api.bcast(
                send_counts_char, empty_int32_arr, root, comm
            )

            # displacements for characters
            displs_char = bodo.ir.join.calc_disp(send_counts_char)

            # compute send counts for nulls
            send_counts_nulls = np.empty(n_pes, np.int64)
            for i in range(n_pes):
                send_counts_nulls[i] = (send_counts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(send_counts_nulls)

            # allocate output with total number of receive elements on this PE
            n_loc = 0 if (is_intercomm and is_sender) else send_counts[rank]
            n_loc_char = 0 if (is_intercomm and is_sender) else send_counts_char[rank]
            recv_arr = alloc_fn(n_loc, n_loc_char)

            # ----- string lengths -----------

            recv_lens = np.empty(n_loc, np.uint32)
            bodo.libs.distributed_impl.c_scatterv(
                send_arr_lens.ctypes,
                send_counts.ctypes,
                displs.ctypes,
                recv_lens.ctypes,
                np.int64(n_loc),
                int32_typ_enum,
                root,
                comm,
            )

            # TODO: don't hardcode offset type. Also, if offset is 32 bit we can
            # use the same buffer
            bodo.libs.str_arr_ext.convert_len_arr_to_offset(
                recv_lens.ctypes, bodo.libs.str_arr_ext.get_offset_ptr(recv_arr), n_loc
            )

            # ----- string characters -----------

            bodo.libs.distributed_impl.c_scatterv(
                bodo.libs.str_arr_ext.get_data_ptr(data),
                send_counts_char.ctypes,
                displs_char.ctypes,
                bodo.libs.str_arr_ext.get_data_ptr(recv_arr),
                np.int64(n_loc_char),
                char_typ_enum,
                root,
                comm,
            )

            # ----------- null bitmap -------------

            n_recv_bytes = (n_loc + 7) >> 3

            send_null_bitmap = bodo.libs.distributed_impl.get_scatter_null_bytes_buff(
                bodo.libs.str_arr_ext.get_null_bitmap_ptr(data),
                send_counts,
                send_counts_nulls,
                is_sender,
            )

            bodo.libs.distributed_impl.c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_nulls.ctypes,
                displs_nulls.ctypes,
                bodo.libs.str_arr_ext.get_null_bitmap_ptr(recv_arr),
                np.int64(n_recv_bytes),
                char_typ_enum,
                root,
                comm,
            )

            return recv_arr

        return impl

    if isinstance(data, ArrayItemArrayType):
        # Code adapted from the string code. Both the string and array(item) codes should be
        # refactored.
        int32_typ_enum = np.int32(numba_to_c_type(types.int32))
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))
        empty_int32_arr = np.array([], np.int32)

        def scatterv_array_item_impl(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            in_offsets_arr = bodo.libs.array_item_arr_ext.get_offsets(data)
            in_data_arr = bodo.libs.array_item_arr_ext.get_data(data)
            in_data_arr = in_data_arr[: in_offsets_arr[-1]]
            in_null_bitmap_arr = bodo.libs.array_item_arr_ext.get_null_bitmap(data)

            rank, is_intercomm, is_sender, n_pes = get_scatter_comm_info(root, comm)
            n_all = bcast_scalar(len(data), root, comm)

            # convert offsets to lengths of lists
            send_arr_lens = np.empty(
                len(data), np.uint32
            )  # XXX offset type is offset_type
            for i in range(len(data)):
                send_arr_lens[i] = in_offsets_arr[i + 1] - in_offsets_arr[i]

            # ------- calculate buffer counts -------

            send_counts = _get_scatterv_send_counts(send_counts, n_pes, n_all)

            # displacements
            displs = bodo.ir.join.calc_disp(send_counts)

            # compute send counts for items
            send_counts_item = np.empty(n_pes, np.int64)
            if is_sender:
                curr_item = 0
                for i in range(n_pes):
                    c = 0
                    for _ in range(send_counts[i]):
                        c += send_arr_lens[curr_item]
                        curr_item += 1
                    send_counts_item[i] = c

            send_counts_item = bodo.libs.distributed_api.bcast(
                send_counts_item, empty_int32_arr, root, comm
            )

            # compute send counts for nulls
            send_counts_nulls = np.empty(n_pes, np.int64)
            for i in range(n_pes):
                send_counts_nulls[i] = (send_counts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(send_counts_nulls)

            # allocate output with total number of receive elements on this PE
            n_loc = 0 if (is_intercomm and is_sender) else send_counts[rank]
            recv_offsets_arr = np.empty(n_loc + 1, np_offset_type)

            recv_data_arr = bodo.libs.distributed_impl.scatterv_impl(
                in_data_arr, send_counts_item, warn_if_dist, root, comm
            )
            n_recv_null_bytes = (n_loc + 7) >> 3
            recv_null_bitmap_arr = np.empty(n_recv_null_bytes, np.uint8)

            # ----- list of item lengths -----------

            recv_lens = np.empty(n_loc, np.uint32)
            c_scatterv(
                send_arr_lens.ctypes,
                send_counts.ctypes,
                displs.ctypes,
                recv_lens.ctypes,
                np.int64(n_loc),
                int32_typ_enum,
                root,
                comm,
            )

            # TODO: don't hardcode offset type. Also, if offset is 32 bit we can
            # use the same buffer
            convert_len_arr_to_offset(recv_lens.ctypes, recv_offsets_arr.ctypes, n_loc)

            # ----------- null bitmap -------------

            send_null_bitmap = get_scatter_null_bytes_buff(
                in_null_bitmap_arr.ctypes, send_counts, send_counts_nulls, is_sender
            )

            c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_nulls.ctypes,
                displs_nulls.ctypes,
                recv_null_bitmap_arr.ctypes,
                np.int64(n_recv_null_bytes),
                char_typ_enum,
                root,
                comm,
            )

            return bodo.libs.array_item_arr_ext.init_array_item_array(
                n_loc, recv_data_arr, recv_offsets_arr, recv_null_bitmap_arr
            )

        return scatterv_array_item_impl

    if data == boolean_array_type:
        # Nullable booleans need their own implementation because the
        # data array stores 1 bit per boolean. As a result, the counts may split
        # may split the data array mid-byte, so we need to handle it the same
        # way we handle the null bitmap. The send count also doesn't reflect the
        # number of bytes to send, so we need to calculate that separately.
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def scatterv_impl_bool_arr(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank, is_intercomm, is_sender, n_pes = get_scatter_comm_info(root, comm)
            data_in = data._data
            null_bitmap = data._null_bitmap
            # Calculate the displacements for nulls and data, each of
            # which is a single bit.
            n_in = len(data)
            n_all = bcast_scalar(n_in, root, comm)

            send_counts = _get_scatterv_send_counts(send_counts, n_pes, n_all)
            # Calculate number of local output elements
            n_loc = np.int64(0 if (is_intercomm and is_sender) else send_counts[rank])
            # compute send counts bytes
            send_counts_bytes = np.empty(n_pes, np.int64)
            for i in range(n_pes):
                send_counts_bytes[i] = (send_counts[i] + 7) >> 3

            displs_bytes = bodo.ir.join.calc_disp(send_counts_bytes)

            send_data_bitmap = get_scatter_null_bytes_buff(
                data_in.ctypes, send_counts, send_counts_bytes, is_sender
            )
            send_null_bitmap = get_scatter_null_bytes_buff(
                null_bitmap.ctypes, send_counts, send_counts_bytes, is_sender
            )
            # Allocate the output arrays
            n_recv_bytes = (
                0 if (is_intercomm and is_sender) else send_counts_bytes[rank]
            )
            data_recv = np.empty(n_recv_bytes, np.uint8)
            bitmap_recv = np.empty(n_recv_bytes, np.uint8)

            c_scatterv(
                send_data_bitmap.ctypes,
                send_counts_bytes.ctypes,
                displs_bytes.ctypes,
                data_recv.ctypes,
                np.int64(n_recv_bytes),
                char_typ_enum,
                root,
                comm,
            )
            c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_bytes.ctypes,
                displs_bytes.ctypes,
                bitmap_recv.ctypes,
                np.int64(n_recv_bytes),
                char_typ_enum,
                root,
                comm,
            )
            return bodo.libs.bool_arr_ext.init_bool_array(data_recv, bitmap_recv, n_loc)

        return scatterv_impl_bool_arr

    if isinstance(
        data,
        (
            IntegerArrayType,
            FloatingArrayType,
            DecimalArrayType,
            DatetimeArrayType,
            TimeArrayType,
        ),
    ) or data in (datetime_date_array_type, timedelta_array_type):
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        # these array need a data array and a null bitmap array to be initialized by
        # their init functions
        if isinstance(data, IntegerArrayType):
            init_func = bodo.libs.int_arr_ext.init_integer_array
        if isinstance(data, FloatingArrayType):
            init_func = bodo.libs.float_arr_ext.init_float_array
        if isinstance(data, DecimalArrayType):
            precision = data.precision
            scale = data.scale
            init_func = numba.njit(no_cpython_wrapper=True)(
                lambda d, b: bodo.libs.decimal_arr_ext.init_decimal_array(
                    d, b, precision, scale
                )  # pragma: no cover
            )
        if isinstance(data, DatetimeArrayType):
            tz = data.tz
            init_func = numba.njit(no_cpython_wrapper=True)(
                lambda d, b: bodo.libs.pd_datetime_arr_ext.init_datetime_array(d, b, tz)
            )  # pragma: no cover
        if data == datetime_date_array_type:
            init_func = bodo.hiframes.datetime_date_ext.init_datetime_date_array
        if data == timedelta_array_type:
            init_func = (
                bodo.hiframes.datetime_timedelta_ext.init_datetime_timedelta_array
            )
        if isinstance(data, TimeArrayType):
            precision = data.precision
            init_func = numba.njit(no_cpython_wrapper=True)(
                lambda d, b: bodo.hiframes.time_ext.init_time_array(
                    d, b, precision
                )  # pragma: no cover
            )

        def scatterv_impl_int_arr(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            data_in = data._data
            null_bitmap = data._null_bitmap
            n_in = len(data_in)

            data_recv = _scatterv_np(data_in, send_counts, warn_if_dist, root, comm)
            out_null_bitmap = _scatterv_null_bitmap(
                null_bitmap, send_counts, n_in, root, comm
            )

            return init_func(data_recv, out_null_bitmap)

        return scatterv_impl_int_arr

    # interval array
    if isinstance(data, IntervalArrayType):
        # scatter the left/right arrays
        def impl_interval_arr(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            left_chunk = bodo.libs.distributed_impl.scatterv_impl(
                data._left, send_counts, warn_if_dist, root, comm
            )
            right_chunk = bodo.libs.distributed_impl.scatterv_impl(
                data._right, send_counts, warn_if_dist, root, comm
            )
            return bodo.libs.interval_arr_ext.init_interval_array(
                left_chunk, right_chunk
            )

        return impl_interval_arr

    # NullArray
    if data == bodo.types.null_array_type:

        def impl_null_arr(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            _, is_intercomm, is_sender, _ = get_scatter_comm_info(root, comm)
            n = bodo.libs.distributed_api.get_node_portion(
                bcast_scalar(len(data), root, comm), bodo.get_size(), bodo.get_rank()
            )
            if is_intercomm and is_sender:
                n = 0
            return bodo.libs.null_arr_ext.init_null_array(n)

        return impl_null_arr

    # TimestampTZ array
    if data == bodo.types.timestamptz_array_type:
        char_typ_enum = np.int32(numba_to_c_type(types.uint8))

        def impl_timestamp_tz_arr(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            _, _, is_sender, n_pes = get_scatter_comm_info(root, comm)

            data_ts_in = data.data_ts
            data_offset_in = data.data_offset
            null_bitmap = data._null_bitmap
            n_in = len(data_ts_in)

            data_ts_recv = _scatterv_np(
                data_ts_in, send_counts, warn_if_dist, root, comm
            )
            data_offset_recv = _scatterv_np(
                data_offset_in, send_counts, warn_if_dist, root, comm
            )

            n_all = bcast_scalar(n_in, root, comm)
            n_recv_bytes = (len(data_ts_recv) + 7) >> 3
            bitmap_recv = np.empty(n_recv_bytes, np.uint8)

            send_counts = _get_scatterv_send_counts(send_counts, n_pes, n_all)

            # compute send counts for nulls
            send_counts_nulls = np.empty(n_pes, np.int64)
            for i in range(n_pes):
                send_counts_nulls[i] = (send_counts[i] + 7) >> 3

            # displacements for nulls
            displs_nulls = bodo.ir.join.calc_disp(send_counts_nulls)

            send_null_bitmap = get_scatter_null_bytes_buff(
                null_bitmap.ctypes, send_counts, send_counts_nulls, is_sender
            )

            c_scatterv(
                send_null_bitmap.ctypes,
                send_counts_nulls.ctypes,
                displs_nulls.ctypes,
                bitmap_recv.ctypes,
                np.int64(n_recv_bytes),
                char_typ_enum,
                root,
                comm,
            )
            return bodo.hiframes.timestamptz_ext.init_timestamptz_array(
                data_ts_recv, data_offset_recv, bitmap_recv
            )

        return impl_timestamp_tz_arr

    if isinstance(data, bodo.hiframes.pd_index_ext.RangeIndexType):
        # TODO: support send_counts
        def impl_range_index(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank, is_intercomm, is_sender, n_pes = get_scatter_comm_info(root, comm)

            start = data._start
            stop = data._stop
            step = data._step
            name = data._name

            name = bcast_scalar(name, root, comm)

            start = bcast_scalar(start, root, comm)
            stop = bcast_scalar(stop, root, comm)
            step = bcast_scalar(step, root, comm)
            n_items = bodo.libs.array_kernels.calc_nitems(start, stop, step)
            chunk_start = bodo.libs.distributed_api.get_start(n_items, n_pes, rank)
            chunk_count = bodo.libs.distributed_api.get_node_portion(
                n_items, n_pes, rank
            )
            new_start = start + step * chunk_start
            new_stop = start + step * (chunk_start + chunk_count)
            new_stop = min(new_stop, stop) if step > 0 else max(new_stop, stop)

            if is_intercomm and is_sender:
                new_start = new_stop = 0

            return bodo.hiframes.pd_index_ext.init_range_index(
                new_start, new_stop, step, name
            )

        return impl_range_index

    # Period index requires special handling because index_from_array
    # doesn't work properly (can't infer the index).
    # See [BE-2067]
    if isinstance(data, bodo.hiframes.pd_index_ext.PeriodIndexType):
        freq = data.freq

        def impl_period_index(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            data_in = data._data
            name = data._name
            name = bcast_scalar(name, root, comm)
            arr = bodo.libs.distributed_impl.scatterv_impl(
                data_in, send_counts, warn_if_dist, root, comm
            )
            return bodo.hiframes.pd_index_ext.init_period_index(arr, name, freq)

        return impl_period_index

    if bodo.hiframes.pd_index_ext.is_pd_index_type(data):

        def impl_pd_index(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            data_in = data._data
            name = data._name
            name = bcast_scalar(name, root, comm)
            arr = bodo.libs.distributed_impl.scatterv_impl(
                data_in, send_counts, warn_if_dist, root, comm
            )
            return bodo.utils.conversion.index_from_array(arr, name)

        return impl_pd_index

    # MultiIndex index
    if isinstance(data, bodo.hiframes.pd_multi_index_ext.MultiIndexType):
        # TODO: handle `levels` and `codes` when available
        def impl_multi_index(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            all_data = bodo.libs.distributed_impl.scatterv_impl(
                data._data, send_counts, warn_if_dist, root, comm
            )
            name = bcast_scalar(data._name, root, comm)
            names = bcast_tuple(data._names, root, comm)
            return bodo.hiframes.pd_multi_index_ext.init_multi_index(
                all_data, names, name
            )

        return impl_multi_index

    if isinstance(data, bodo.hiframes.pd_series_ext.SeriesType):

        def impl_series(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            # get data and index arrays
            arr = bodo.hiframes.pd_series_ext.get_series_data(data)
            index = bodo.hiframes.pd_series_ext.get_series_index(data)
            name = bodo.hiframes.pd_series_ext.get_series_name(data)
            # scatter data
            out_name = bcast_scalar(name, root, comm)
            out_arr = bodo.libs.distributed_impl.scatterv_impl(
                arr, send_counts, warn_if_dist, root, comm
            )
            out_index = bodo.libs.distributed_impl.scatterv_impl(
                index, send_counts, warn_if_dist, root, comm
            )
            # create output Series
            return bodo.hiframes.pd_series_ext.init_series(out_arr, out_index, out_name)

        return impl_series

    if isinstance(data, bodo.hiframes.pd_dataframe_ext.DataFrameType):
        n_cols = len(data.columns)
        __col_name_meta_scaterv_impl = ColNamesMetaType(data.columns)

        func_text = f"def impl_df(data, send_counts=None, warn_if_dist=True, root={DEFAULT_ROOT}, comm=0):\n"
        if data.is_table_format:
            func_text += (
                "  table = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(data)\n"
            )
            func_text += "  g_table = bodo.libs.distributed_impl.scatterv_impl(table, send_counts, warn_if_dist, root, comm)\n"
            data_args = "g_table"
        else:
            for i in range(n_cols):
                func_text += f"  data_{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(data, {i})\n"
                func_text += f"  g_data_{i} = bodo.libs.distributed_impl.scatterv_impl(data_{i}, send_counts, warn_if_dist, root, comm)\n"
            data_args = ", ".join(f"g_data_{i}" for i in range(n_cols))
        func_text += (
            "  index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(data)\n"
        )
        func_text += "  g_index = bodo.libs.distributed_impl.scatterv_impl(index, send_counts, warn_if_dist, root, comm)\n"
        func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_args},), g_index, __col_name_meta_scaterv_impl)\n"
        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "__col_name_meta_scaterv_impl": __col_name_meta_scaterv_impl,
            },
            loc_vars,
        )
        impl_df = loc_vars["impl_df"]
        return impl_df

    if isinstance(data, bodo.types.TableType):
        func_text = f"def impl_table(data, send_counts=None, warn_if_dist=True, root={DEFAULT_ROOT}, comm=0):\n"
        func_text += "  T = data\n"
        func_text += "  T2 = init_table(T, False)\n"
        func_text += "  l = 0\n"

        glbls = {}
        for blk in data.type_to_blk.values():
            glbls[f"arr_inds_{blk}"] = np.array(
                data.block_to_arr_ind[blk], dtype=np.int64
            )
            func_text += f"  arr_list_{blk} = get_table_block(T, {blk})\n"
            func_text += f"  out_arr_list_{blk} = alloc_list_like(arr_list_{blk}, len(arr_list_{blk}), False)\n"
            func_text += f"  for i in range(len(arr_list_{blk})):\n"
            func_text += f"    arr_ind_{blk} = arr_inds_{blk}[i]\n"
            func_text += (
                f"    ensure_column_unboxed(T, arr_list_{blk}, i, arr_ind_{blk})\n"
            )
            func_text += f"    out_arr_{blk} = bodo.libs.distributed_impl.scatterv_impl(arr_list_{blk}[i], send_counts, warn_if_dist, root, comm)\n"
            func_text += f"    out_arr_list_{blk}[i] = out_arr_{blk}\n"
            func_text += f"    l = len(out_arr_{blk})\n"
            func_text += f"  T2 = set_table_block(T2, out_arr_list_{blk}, {blk})\n"
        func_text += "  T2 = set_table_len(T2, l)\n"
        func_text += "  return T2\n"

        glbls.update(
            {
                "bodo": bodo,
                "init_table": bodo.hiframes.table.init_table,
                "get_table_block": bodo.hiframes.table.get_table_block,
                "ensure_column_unboxed": bodo.hiframes.table.ensure_column_unboxed,
                "set_table_block": bodo.hiframes.table.set_table_block,
                "set_table_len": bodo.hiframes.table.set_table_len,
                "alloc_list_like": bodo.hiframes.table.alloc_list_like,
            }
        )
        loc_vars = {}
        exec(func_text, glbls, loc_vars)
        return loc_vars["impl_table"]

    if data == bodo.types.dict_str_arr_type:
        empty_int32_arr = np.array([], np.int32)

        def impl_dict_arr(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            # broadcast the dictionary data (string array)
            str_arr = bodo.libs.distributed_api.bcast(
                data._data, empty_int32_arr, root, comm
            )
            # scatter indices array
            new_indices = bodo.libs.distributed_impl.scatterv_impl(
                data._indices, send_counts, warn_if_dist, root, comm
            )
            # the dictionary is global by construction (broadcast)
            return bodo.libs.dict_arr_ext.init_dict_arr(
                str_arr, new_indices, True, data._has_unique_local_dictionary, None
            )

        return impl_dict_arr

    if isinstance(data, CategoricalArrayType):

        def impl_cat(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            codes = bodo.libs.distributed_impl.scatterv_impl(
                data.codes, send_counts, warn_if_dist, root, comm
            )
            return bodo.hiframes.pd_categorical_ext.init_categorical_array(
                codes, data.dtype
            )

        return impl_cat

    # Tuple of data containers
    if isinstance(data, types.BaseTuple):
        func_text = f"def impl_tuple(data, send_counts=None, warn_if_dist=True, root={DEFAULT_ROOT}, comm=0):\n"
        func_text += "  return ({}{})\n".format(
            ", ".join(
                f"bodo.libs.distributed_impl.scatterv_impl(data[{i}], send_counts, warn_if_dist, root, comm)"
                for i in range(len(data))
            ),
            "," if len(data) > 0 else "",
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        impl_tuple = loc_vars["impl_tuple"]
        return impl_tuple

    # List of distributable data
    if isinstance(data, types.List) and is_distributable_typ(data.dtype):

        def impl_list(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            is_sender = rank == root
            if comm != 0:
                is_sender = root == MPI.ROOT

            length = bcast_scalar(len(data), root, comm)
            out = []
            for i in range(length):
                in_val = data[i] if is_sender else data[0]
                out.append(
                    bodo.libs.distributed_impl.scatterv_impl(
                        in_val, send_counts, warn_if_dist, root, comm
                    )
                )

            return out

        return impl_list

    # Dictionary of distributable data
    if isinstance(data, types.DictType) and is_distributable_typ(data.value_type):

        def impl_dict(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            rank = bodo.libs.distributed_api.get_rank()
            is_sender = rank == root
            if comm != 0:
                is_sender = root == MPI.ROOT

            length = bcast_scalar(len(data), root, comm)
            in_keys = list(data.keys())
            in_values = list(data.values())
            out = {}
            for i in range(length):
                key = in_keys[i] if is_sender else in_keys[0]
                value = in_values[i] if is_sender else in_values[0]
                out_key = bcast_scalar(key, root, comm)
                out[out_key] = bodo.libs.distributed_impl.scatterv_impl(
                    value, send_counts, warn_if_dist, root, comm
                )

            return out

        return impl_dict

    # StructArray
    if isinstance(data, bodo.types.StructArrayType):
        n_fields = len(data.data)
        func_text = f"def impl_struct(data, send_counts=None, warn_if_dist=True, root={DEFAULT_ROOT}, comm=0):\n"
        func_text += "  inner_data_arrs = bodo.libs.struct_arr_ext.get_data(data)\n"
        func_text += "  out_null_bitmap = _scatterv_null_bitmap(bodo.libs.struct_arr_ext.get_null_bitmap(data), send_counts, len(data), root, comm)\n"
        for i in range(n_fields):
            func_text += f"  new_inner_data_arr_{i} = bodo.libs.distributed_impl.scatterv_impl(inner_data_arrs[{i}], send_counts, warn_if_dist, root, comm)\n"

        new_data_tuple_str = "({}{})".format(
            ", ".join([f"new_inner_data_arr_{i}" for i in range(n_fields)]),
            "," if n_fields > 0 else "",
        )
        field_names_tuple_str = "({}{})".format(
            ", ".join([f"'{f}'" for f in data.names]),
            "," if n_fields > 0 else "",
        )
        out_len = (
            "len(new_inner_data_arr_0)"
            if n_fields > 0
            else "bodo.libs.distributed_api.get_node_portion(bcast_scalar(len(data), root, comm), bodo.get_size(), bodo.get_rank())"
        )
        func_text += f"  return bodo.libs.struct_arr_ext.init_struct_arr({out_len}, {new_data_tuple_str}, out_null_bitmap, {field_names_tuple_str})\n"
        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "_scatterv_null_bitmap": _scatterv_null_bitmap,
                "bcast_scalar": bcast_scalar,
            },
            loc_vars,
        )
        impl_struct = loc_vars["impl_struct"]
        return impl_struct

    # MapArrayType
    if isinstance(data, bodo.types.MapArrayType):

        def impl(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            # Call it recursively on the underlying ArrayItemArray array.
            new_underlying_data = bodo.libs.distributed_impl.scatterv_impl(
                data._data, send_counts, warn_if_dist, root, comm
            )
            # Reconstruct the Map array from the new ArrayItemArray array.
            new_data = bodo.libs.map_arr_ext.init_map_arr(new_underlying_data)
            return new_data

        return impl

    # TupleArray
    if isinstance(data, bodo.types.TupleArrayType):

        def impl_tuple(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            new_underlying_data = bodo.libs.distributed_impl.scatterv_impl(
                data._data, send_counts, warn_if_dist, root, comm
            )
            return bodo.libs.tuple_arr_ext.init_tuple_arr(new_underlying_data)

        return impl_tuple

    if data is types.none:  # pragma: no cover
        return (
            lambda data,
            send_counts=None,
            warn_if_dist=True,
            root=DEFAULT_ROOT,
            comm=0: None
        )

    if isinstance(data, bodo.types.MatrixType):

        def impl_matrix(
            data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
        ):  # pragma: no cover
            new_underlying_data = bodo.libs.distributed_impl.scatterv_impl(
                data.data, send_counts, warn_if_dist, root, comm
            )
            return bodo.libs.matrix_ext.init_np_matrix(new_underlying_data)

        return impl_matrix

    raise BodoError(f"scatterv() not available for {data}")  # pragma: no cover


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def _scatterv_np(data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0):
    """scatterv() implementation for numpy arrays, refactored here with
    no_cpython_wrapper=True to enable int128 data array of decimal arrays. Otherwise,
    Numba creates a wrapper and complains about unboxing int128.
    """
    from bodo.libs.distributed_api import bcast_tuple, get_tuple_prod

    typ_val = numba_to_c_type(data.dtype)
    ndim = data.ndim
    dtype = data.dtype
    # using np.dtype since empty() doesn't work with typeref[datetime/timedelta]
    if dtype == types.NPDatetime("ns"):
        dtype = np.dtype("datetime64[ns]")
    elif dtype == types.NPTimedelta("ns"):
        dtype = np.dtype("timedelta64[ns]")
    zero_shape = (0,) * ndim

    def scatterv_arr_impl(
        data, send_counts=None, warn_if_dist=True, root=DEFAULT_ROOT, comm=0
    ):  # pragma: no cover
        rank, is_intercomm, is_sender, n_pes = get_scatter_comm_info(root, comm)

        data_in = np.ascontiguousarray(data)
        data_ptr = data_in.ctypes

        # broadcast shape to all processors
        shape = zero_shape
        if is_sender:
            shape = data_in.shape
        shape = bcast_tuple(shape, root, comm)
        n_elem_per_row = get_tuple_prod(shape[1:])

        send_counts = _get_scatterv_send_counts(send_counts, n_pes, shape[0])
        send_counts *= n_elem_per_row

        # allocate output with total number of receive elements on this PE
        n_loc = 0 if (is_intercomm and is_sender) else send_counts[rank]
        recv_data = np.empty(n_loc, dtype)

        # displacements
        displs = bodo.ir.join.calc_disp(send_counts)

        c_scatterv(
            data_ptr,
            send_counts.ctypes,
            displs.ctypes,
            recv_data.ctypes,
            np.int64(n_loc),
            np.int32(typ_val),
            root,
            comm,
        )

        if is_intercomm and is_sender:
            shape = zero_shape

        # handle multi-dim case
        return recv_data.reshape((-1,) + shape[1:])

    return scatterv_arr_impl


char_typ_enum = np.int32(numba_to_c_type(types.uint8))


@numba.njit(cache=True, no_cpython_wrapper=True)
def _scatterv_null_bitmap(null_bitmap, send_counts, n_in, root, comm):
    """Scatter null bitmap for nullable arrays"""
    rank, is_intercomm, is_sender, n_pes = get_scatter_comm_info(root, comm)

    n_all = bodo.libs.distributed_api.bcast_scalar(n_in, root, comm)

    send_counts = _get_scatterv_send_counts(send_counts, n_pes, n_all)
    n_loc = 0 if (is_intercomm and is_sender) else send_counts[rank]

    n_recv_bytes = (n_loc + 7) >> 3
    bitmap_recv = np.empty(n_recv_bytes, np.uint8)

    # compute send counts for nulls
    send_counts_nulls = np.empty(n_pes, np.int64)
    for i in range(n_pes):
        send_counts_nulls[i] = (send_counts[i] + 7) >> 3

    # displacements for nulls
    displs_nulls = bodo.ir.join.calc_disp(send_counts_nulls)

    send_null_bitmap = get_scatter_null_bytes_buff(
        null_bitmap.ctypes, send_counts, send_counts_nulls, is_sender
    )

    c_scatterv(
        send_null_bitmap.ctypes,
        send_counts_nulls.ctypes,
        displs_nulls.ctypes,
        bitmap_recv.ctypes,
        np.int64(n_recv_bytes),
        char_typ_enum,
        root,
        comm,
    )
    return bitmap_recv


@numba.njit(cache=True, no_cpython_wrapper=True)
def get_scatter_comm_info(root, comm):
    """Return communication attributes for scatterv based on root and intercomm"""
    is_intercomm = comm != 0
    rank = bodo.libs.distributed_api.get_rank()
    is_sender = rank == root
    if is_intercomm:
        is_sender = root == MPI.ROOT
    n_pes = (
        bodo.libs.distributed_api.get_size()
        if not (is_intercomm and is_sender)
        else bodo.libs.distributed_api.get_remote_size(comm)
    )
    return rank, is_intercomm, is_sender, n_pes


@numba.njit(cache=True)
def get_scatter_null_bytes_buff(
    null_bitmap_ptr, sendcounts, sendcounts_nulls, is_sender
):  # pragma: no cover
    """copy null bytes into a padded buffer for scatter.
    Padding is needed since processors receive whole bytes and data inside border bytes
    has to be split.
    Only the root rank has the input data and needs to create a valid send buffer.
    """
    # non-root ranks don't have scatter input
    if not is_sender:
        return np.empty(1, np.uint8)

    null_bytes_buff = np.empty(sendcounts_nulls.sum(), np.uint8)

    curr_tmp_byte = 0  # current location in scatter buffer
    curr_str = 0  # current string in input bitmap

    # for each rank
    for i_rank in range(len(sendcounts)):
        n_strs = sendcounts[i_rank]
        n_bytes = sendcounts_nulls[i_rank]
        chunk_bytes = null_bytes_buff[curr_tmp_byte : curr_tmp_byte + n_bytes]
        # for each string in chunk
        for j in range(n_strs):
            set_bit_to_arr(chunk_bytes, j, get_bit_bitmap(null_bitmap_ptr, curr_str))
            curr_str += 1

        curr_tmp_byte += n_bytes

    return null_bytes_buff


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def _get_scatterv_send_counts(send_counts, n_pes, n):
    """compute send counts if 'send_counts' is None."""
    if not is_overload_none(send_counts):
        return lambda send_counts, n_pes, n: send_counts

    def impl(send_counts, n_pes, n):  # pragma: no cover
        # compute send counts if not available
        send_counts = np.empty(n_pes, np.int64)
        for i in range(n_pes):
            send_counts[i] = bodo.libs.distributed_api.get_node_portion(n, n_pes, i)
        return send_counts

    return impl


def irecv_impl(arr, size, pe, tag, cond):
    """Implementation for distributed_api.irecv()"""
    from bodo.libs import hdist
    from bodo.libs.distributed_api import get_type_enum

    mpi_req_numba_type = getattr(types, "int" + str(8 * hdist.mpi_req_num_bytes))

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
        or arr == bodo.types.timedelta_array_type
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
