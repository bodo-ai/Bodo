"""implementations of rolling window functions (sequential and parallel)"""

import numba
import numpy as np
import pandas as pd
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed
from numba.core.typing import signature
from numba.core.typing.templates import AbstractTemplate, infer_global
from numba.extending import lower_builtin, overload, register_jitable

import bodo
from bodo.libs.distributed_api import Reduce_Type
from bodo.utils.typing import (
    BodoError,
    assert_bodo_error,
    decode_if_dict_array,
    get_overload_const_func,
    get_overload_const_str,
    is_const_func_type,
    is_overload_constant_bool,
    is_overload_constant_str,
    is_overload_none,
    is_overload_true,
)
from bodo.utils.utils import unliteral_all

supported_rolling_funcs = (
    "sum",
    "mean",
    "var",
    "std",
    "count",
    "median",
    "min",
    "max",
    "cov",
    "corr",
    "apply",
)


unsupported_rolling_methods = [
    "skew",
    "kurt",
    "aggregate",
    "quantile",
    "sem",
]


def rolling_fixed(arr, win):  # pragma: no cover
    return arr


def rolling_variable(arr, on_arr, win):  # pragma: no cover
    return arr


def rolling_cov(arr, arr2, win):  # pragma: no cover
    return arr


def rolling_corr(arr, arr2, win):  # pragma: no cover
    return arr


@infer_global(rolling_cov)
@infer_global(rolling_corr)
class RollingCovType(AbstractTemplate):
    def generic(self, args, kws):
        arr = args[0]  # array

        ret_typ = arr.copy(dtype=types.float64)
        return signature(ret_typ, *unliteral_all(args))


@lower_builtin(rolling_corr, types.VarArg(types.Any))
@lower_builtin(rolling_cov, types.VarArg(types.Any))
def lower_rolling_corr_dummy(context, builder, sig, args):
    return context.get_constant_null(sig.return_type)


@overload(rolling_fixed, no_unliteral=True)
def overload_rolling_fixed(
    arr, index_arr, win, minp, center, fname, raw=True, parallel=False
):
    assert_bodo_error(
        is_overload_constant_bool(raw), "raw argument should be constant bool"
    )
    # UDF case
    if is_const_func_type(fname):
        func = _get_apply_func(fname)
        return (
            lambda arr,
            index_arr,
            win,
            minp,
            center,
            fname,
            raw=True,
            parallel=False: roll_fixed_apply(
                arr, index_arr, win, minp, center, parallel, func, raw
            )
        )  # pragma: no cover

    assert_bodo_error(is_overload_constant_str(fname))
    func_name = get_overload_const_str(fname)

    if func_name not in ("sum", "mean", "var", "std", "count", "median", "min", "max"):
        raise BodoError(f"invalid rolling (fixed window) function {func_name}")

    if func_name in ("median", "min", "max"):
        # just using 'apply' since we don't have streaming/linear support
        # TODO: implement linear support similar to others
        func_text = "def kernel_func(A):\n"
        func_text += "  if np.isnan(A).sum() != 0: return np.nan\n"
        func_text += f"  return np.{func_name}(A)\n"
        loc_vars = {}
        exec(func_text, {"np": np}, loc_vars)
        # We can't use numba.njit because it generates a CPUDispatcher which
        # in the case of kernel_func gets passed as argument to other functions,
        # in the form of a dynamic global address that prevents caching.
        # With register_jitable a dummy value is passed instead, and numba
        # knows which function call to insert in the library
        kernel_func = register_jitable(loc_vars["kernel_func"])

        return (
            lambda arr,
            index_arr,
            win,
            minp,
            center,
            fname,
            raw=True,
            parallel=False: roll_fixed_apply(
                arr, index_arr, win, minp, center, parallel, kernel_func
            )
        )  # pragma: no cover

    init_kernel, add_kernel, remove_kernel, calc_kernel = linear_kernels[func_name]
    return (
        lambda arr,
        index_arr,
        win,
        minp,
        center,
        fname,
        raw=True,
        parallel=False: roll_fixed_linear_generic(
            arr,
            win,
            minp,
            center,
            parallel,
            init_kernel,
            add_kernel,
            remove_kernel,
            calc_kernel,
        )
    )  # pragma: no cover


@overload(rolling_variable, no_unliteral=True)
def overload_rolling_variable(
    arr, on_arr, index_arr, win, minp, center, fname, raw=True, parallel=False
):
    assert_bodo_error(is_overload_constant_bool(raw))
    # UDF case
    if is_const_func_type(fname):
        func = _get_apply_func(fname)
        return (
            lambda arr,
            on_arr,
            index_arr,
            win,
            minp,
            center,
            fname,
            raw=True,
            parallel=False: roll_variable_apply(
                arr, on_arr, index_arr, win, minp, center, parallel, func, raw
            )
        )  # pragma: no cover

    assert_bodo_error(is_overload_constant_str(fname))
    func_name = get_overload_const_str(fname)

    if func_name not in ("sum", "mean", "var", "std", "count", "median", "min", "max"):
        raise BodoError(f"invalid rolling (variable window) function {func_name}")

    if func_name in ("median", "min", "max"):
        # just using 'apply' since we don't have streaming/linear support
        # TODO: implement linear support similar to others
        func_text = "def kernel_func(A):\n"
        func_text += "  arr  = dropna(A)\n"
        func_text += "  if len(arr) == 0: return np.nan\n"
        func_text += f"  return np.{func_name}(arr)\n"
        loc_vars = {}
        exec(func_text, {"np": np, "dropna": _dropna}, loc_vars)
        # We can't use numba.njit because it generates a CPUDispatcher which
        # in the case of kernel_func gets passed as argument to other functions,
        # in the form of a dynamic global address that prevents caching.
        # With register_jitable a dummy value is passed instead, and numba
        # knows which function call to insert in the library
        kernel_func = register_jitable(loc_vars["kernel_func"])

        return (
            lambda arr,
            on_arr,
            index_arr,
            win,
            minp,
            center,
            fname,
            raw=True,
            parallel=False: roll_variable_apply(
                arr, on_arr, index_arr, win, minp, center, parallel, kernel_func
            )
        )  # pragma: no cover

    init_kernel, add_kernel, remove_kernel, calc_kernel = linear_kernels[func_name]
    return (
        lambda arr,
        on_arr,
        index_arr,
        win,
        minp,
        center,
        fname,
        raw=True,
        parallel=False: roll_var_linear_generic(
            arr,
            on_arr,
            win,
            minp,
            center,
            parallel,
            init_kernel,
            add_kernel,
            remove_kernel,
            calc_kernel,
        )
    )  # pragma: no cover


def _get_apply_func(f_type):
    """get UDF function from function type and jit it with Bodo's sequential pipeline to
    avoid parallel errors.
    """
    func = get_overload_const_func(f_type, None)
    return bodo.compiler.udf_jit(func)


#### adapted from pandas window.pyx ####

comm_border_tag = 22  # arbitrary, TODO: revisit comm tags


@register_jitable
def roll_fixed_linear_generic(
    in_arr, win, minp, center, parallel, init_data, add_obs, remove_obs, calc_out
):  # pragma: no cover
    _validate_roll_fixed_args(win, minp)

    in_arr = prep_values(in_arr)
    rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    N = len(in_arr)
    offset = (win - 1) // 2 if center else 0

    if parallel:
        # halo length is w/2 to handle even w such as w=4
        halo_size = np.int32(win // 2) if center else np.int32(win - 1)
        if _is_small_for_parallel(N, halo_size):
            return _handle_small_data(
                in_arr,
                win,
                minp,
                center,
                rank,
                n_pes,
                init_data,
                add_obs,
                remove_obs,
                calc_out,
            )

        comm_data = _border_icomm(in_arr, rank, n_pes, halo_size, True, center)
        (
            l_recv_buff,
            r_recv_buff,
            l_send_req,
            r_send_req,
            l_recv_req,
            r_recv_req,
        ) = comm_data

    output, data = roll_fixed_linear_generic_seq(
        in_arr, win, minp, center, init_data, add_obs, remove_obs, calc_out
    )

    if parallel:
        _border_send_wait(r_send_req, l_send_req, rank, n_pes, True, center)

        # recv right
        if center and rank != n_pes - 1:
            bodo.libs.distributed_api.wait(r_recv_req, True)

            for i in range(0, halo_size):
                data = add_obs(r_recv_buff[i], *data)

                prev_x = in_arr[N + i - win]
                data = remove_obs(prev_x, *data)

                output[N + i - offset] = calc_out(minp, *data)

        # recv left
        if rank != 0:
            bodo.libs.distributed_api.wait(l_recv_req, True)
            data = init_data()
            for i in range(0, halo_size):
                data = add_obs(l_recv_buff[i], *data)

            for i in range(0, win - 1):
                data = add_obs(in_arr[i], *data)

                if i > offset:
                    prev_x = l_recv_buff[i - offset - 1]
                    data = remove_obs(prev_x, *data)

                if i >= offset:
                    output[i - offset] = calc_out(minp, *data)

    return output


@register_jitable
def roll_fixed_linear_generic_seq(
    in_arr, win, minp, center, init_data, add_obs, remove_obs, calc_out
):  # pragma: no cover
    data = init_data()
    N = len(in_arr)
    offset = (win - 1) // 2 if center else 0
    output = np.empty(N, dtype=np.float64)
    range_endpoint = max(minp, 1) - 1
    # in case window is smaller than array
    range_endpoint = min(range_endpoint, N)

    for i in range(0, range_endpoint):
        data = add_obs(in_arr[i], *data)
        if i >= offset:
            output[i - offset] = calc_out(minp, *data)

    for i in range(range_endpoint, N):
        val = in_arr[i]
        data = add_obs(val, *data)

        if i > win - 1:
            prev_x = in_arr[i - win]
            data = remove_obs(prev_x, *data)

        output[i - offset] = calc_out(minp, *data)

    border_data = data  # used for parallel case with center=True

    for i in range(N, N + offset):
        if i > win - 1:
            prev_x = in_arr[i - win]
            data = remove_obs(prev_x, *data)

        output[i - offset] = calc_out(minp, *data)

    return output, border_data


def roll_fixed_apply(
    in_arr, index_arr, win, minp, center, parallel, kernel_func, raw=True
):  # pragma: no cover
    pass


@overload(roll_fixed_apply, no_unliteral=True)
def overload_roll_fixed_apply(
    in_arr, index_arr, win, minp, center, parallel, kernel_func, raw=True
):
    assert_bodo_error(is_overload_constant_bool(raw))
    return roll_fixed_apply_impl


def roll_fixed_apply_impl(
    in_arr, index_arr, win, minp, center, parallel, kernel_func, raw=True
):  # pragma: no cover
    _validate_roll_fixed_args(win, minp)

    in_arr = prep_values(in_arr)
    rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    N = len(in_arr)
    offset = (win - 1) // 2 if center else 0
    # replace index_arr=None argument (passed when index_arr is not needed) with dummy
    # array to avoid errors
    index_arr = fix_index_arr(index_arr)

    if parallel:
        # halo length is w/2 to handle even w such as w=4
        halo_size = np.int32(win // 2) if center else np.int32(win - 1)
        if _is_small_for_parallel(N, halo_size):
            return _handle_small_data_apply(
                in_arr, index_arr, win, minp, center, rank, n_pes, kernel_func, raw
            )

        comm_data = _border_icomm(in_arr, rank, n_pes, halo_size, True, center)
        (
            l_recv_buff,
            r_recv_buff,
            l_send_req,
            r_send_req,
            l_recv_req,
            r_recv_req,
        ) = comm_data

        if raw == False:
            comm_data_idx = _border_icomm(
                index_arr, rank, n_pes, halo_size, True, center
            )
            (
                l_recv_buff_idx,
                r_recv_buff_idx,
                l_send_req_idx,
                r_send_req_idx,
                l_recv_req_idx,
                r_recv_req_idx,
            ) = comm_data_idx

    output = roll_fixed_apply_seq(
        in_arr, index_arr, win, minp, center, kernel_func, raw
    )

    if parallel:
        _border_send_wait(r_send_req, l_send_req, rank, n_pes, True, center)
        if raw == False:
            _border_send_wait(r_send_req_idx, l_send_req_idx, rank, n_pes, True, center)

        # recv right
        if center and rank != n_pes - 1:
            bodo.libs.distributed_api.wait(r_recv_req, True)
            if raw == False:
                bodo.libs.distributed_api.wait(r_recv_req_idx, True)

            recv_right_compute(
                output,
                in_arr,
                index_arr,
                N,
                win,
                minp,
                offset,
                r_recv_buff,
                r_recv_buff_idx,
                kernel_func,
                raw,
            )

        # recv left
        if rank != 0:
            bodo.libs.distributed_api.wait(l_recv_req, True)
            if raw == False:
                bodo.libs.distributed_api.wait(l_recv_req_idx, True)

            recv_left_compute(
                output,
                in_arr,
                index_arr,
                win,
                minp,
                offset,
                l_recv_buff,
                l_recv_buff_idx,
                kernel_func,
                raw,
            )

    return output


def recv_right_compute(
    output,
    in_arr,
    index_arr,
    N,
    win,
    minp,
    offset,
    r_recv_buff,
    r_recv_buff_idx,
    kernel_func,
    raw,
):
    pass


@overload(recv_right_compute, no_unliteral=True)
def overload_recv_right_compute(
    output,
    in_arr,
    index_arr,
    N,
    win,
    minp,
    offset,
    r_recv_buff,
    r_recv_buff_idx,
    kernel_func,
    raw,
):
    assert_bodo_error(is_overload_constant_bool(raw))
    if is_overload_true(raw):

        def impl(
            output,
            in_arr,
            index_arr,
            N,
            win,
            minp,
            offset,
            r_recv_buff,
            r_recv_buff_idx,
            kernel_func,
            raw,
        ):
            border_data = np.concatenate((in_arr[N - win + 1 :], r_recv_buff))
            ind = 0
            for i in range(max(N - offset, 0), N):
                data = border_data[ind : ind + win]
                if win - np.isnan(data).sum() < minp:
                    output[i] = np.nan
                else:
                    output[i] = kernel_func(data)
                ind += 1

        return impl

    def impl_series(
        output,
        in_arr,
        index_arr,
        N,
        win,
        minp,
        offset,
        r_recv_buff,
        r_recv_buff_idx,
        kernel_func,
        raw,
    ):
        border_data = np.concatenate((in_arr[N - win + 1 :], r_recv_buff))
        border_data_idx = np.concatenate((index_arr[N - win + 1 :], r_recv_buff_idx))
        ind = 0
        for i in range(max(N - offset, 0), N):
            data = border_data[ind : ind + win]
            if win - np.isnan(data).sum() < minp:
                output[i] = np.nan
            else:
                output[i] = kernel_func(
                    pd.Series(data, border_data_idx[ind : ind + win])
                )
            ind += 1

    return impl_series


def recv_left_compute(
    output,
    in_arr,
    index_arr,
    win,
    minp,
    offset,
    l_recv_buff,
    l_recv_buff_idx,
    kernel_func,
    raw,
):
    pass


@overload(recv_left_compute, no_unliteral=True)
def overload_recv_left_compute(
    output,
    in_arr,
    index_arr,
    win,
    minp,
    offset,
    l_recv_buff,
    l_recv_buff_idx,
    kernel_func,
    raw,
):
    assert_bodo_error(is_overload_constant_bool(raw))
    if is_overload_true(raw):

        def impl(
            output,
            in_arr,
            index_arr,
            win,
            minp,
            offset,
            l_recv_buff,
            l_recv_buff_idx,
            kernel_func,
            raw,
        ):
            border_data = np.concatenate((l_recv_buff, in_arr[: win - 1]))
            for i in range(0, win - offset - 1):
                data = border_data[i : i + win]
                if win - np.isnan(data).sum() < minp:
                    output[i] = np.nan
                else:
                    output[i] = kernel_func(data)

        return impl

    def impl_series(
        output,
        in_arr,
        index_arr,
        win,
        minp,
        offset,
        l_recv_buff,
        l_recv_buff_idx,
        kernel_func,
        raw,
    ):
        border_data = np.concatenate((l_recv_buff, in_arr[: win - 1]))
        border_data_idx = np.concatenate((l_recv_buff_idx, index_arr[: win - 1]))
        for i in range(0, win - offset - 1):
            data = border_data[i : i + win]
            if win - np.isnan(data).sum() < minp:
                output[i] = np.nan
            else:
                output[i] = kernel_func(pd.Series(data, border_data_idx[i : i + win]))

    return impl_series


def roll_fixed_apply_seq(
    in_arr, index_arr, win, minp, center, kernel_func, raw=True
):  # pragma: no cover
    pass


@overload(roll_fixed_apply_seq, no_unliteral=True)
def overload_roll_fixed_apply_seq(
    in_arr, index_arr, win, minp, center, kernel_func, raw=True
):
    assert_bodo_error(is_overload_constant_bool(raw), "'raw' should be constant bool")

    def roll_fixed_apply_seq_impl(
        in_arr, index_arr, win, minp, center, kernel_func, raw=True
    ):  # pragma: no cover
        N = len(in_arr)
        output = np.empty(N, dtype=np.float64)
        offset = (win - 1) // 2 if center else 0

        for i in range(0, N):
            start = max(i - win + 1 + offset, 0)
            end = min(i + 1 + offset, N)
            data = in_arr[start:end]
            # TODO: use np.isfinite() to match some places in Pandas?
            # https://github.com/pandas-dev/pandas/blob/ddc0256f1526ec35d44f675960dee8403bc28e4a/pandas/_libs/window/aggregations.pyx#L1144
            if end - start - np.isnan(data).sum() < minp:
                output[i] = np.nan
            else:
                output[i] = apply_func(kernel_func, data, index_arr, start, end, raw)

        return output

    return roll_fixed_apply_seq_impl


def apply_func(kernel_func, data, index_arr, start, end, raw):  # pragma: no cover
    return kernel_func(data)


@overload(apply_func, no_unliteral=True)
def overload_apply_func(kernel_func, data, index_arr, start, end, raw):
    assert_bodo_error(is_overload_constant_bool(raw), "'raw' should be constant bool")
    if is_overload_true(raw):
        return lambda kernel_func, data, index_arr, start, end, raw: kernel_func(
            data
        )  # pragma: no cover

    return lambda kernel_func, data, index_arr, start, end, raw: kernel_func(
        pd.Series(data, index_arr[start:end])
    )  # pragma: no cover


def fix_index_arr(A):  # pragma: no cover
    return A


@overload(fix_index_arr)
def overload_fix_index_arr(A):
    """return dummy array if A is None, else A"""
    if is_overload_none(A):
        return lambda A: np.zeros(3)  # pragma: no cover
    return lambda A: A  # pragma: no cover


# -----------------------------
# variable window


def get_offset_nanos(w):  # pragma: no cover
    """convert 'w' to offset if possible. Return success code 0 or failure 1."""
    out = status = 0
    try:
        out = pd.tseries.frequencies.to_offset(w).nanos
    except Exception:
        status = 1
    return out, status


def offset_to_nanos(w):  # pragma: no cover
    return w


@overload(offset_to_nanos)
def overload_offset_to_nanos(w):
    """convert offset value to nanos"""
    if isinstance(w, types.Integer):
        return lambda w: w  # pragma: no cover

    def impl(w):  # pragma: no cover
        with numba.objmode(out="int64", status="int64"):
            out, status = get_offset_nanos(w)
        if status != 0:
            raise ValueError("Invalid offset value")
        return out

    return impl


@register_jitable
def roll_var_linear_generic(
    in_arr,
    on_arr_dt,
    win,
    minp,
    center,
    parallel,
    init_data,
    add_obs,
    remove_obs,
    calc_out,
):  # pragma: no cover
    _validate_roll_var_args(minp, center)

    in_arr = prep_values(in_arr)
    win = offset_to_nanos(win)
    rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    on_arr = cast_dt64_arr_to_int(on_arr_dt)
    N = len(in_arr)
    # Pandas is right closed by default, TODO: extend to support arg
    left_closed = False
    right_closed = True

    if parallel:
        if _is_small_for_parallel_variable(on_arr, win):
            return _handle_small_data_variable(
                in_arr,
                on_arr,
                win,
                minp,
                rank,
                n_pes,
                init_data,
                add_obs,
                remove_obs,
                calc_out,
            )

        comm_data = _border_icomm_var(in_arr, on_arr, rank, n_pes, win)
        (
            l_recv_buff,
            l_recv_t_buff,
            r_send_req,
            r_send_t_req,
            l_recv_req,
            l_recv_t_req,
        ) = comm_data

    start, end = _build_indexer(on_arr, N, win, left_closed, right_closed)
    output = roll_var_linear_generic_seq(
        in_arr, on_arr, win, minp, start, end, init_data, add_obs, remove_obs, calc_out
    )

    if parallel:
        _border_send_wait(r_send_req, r_send_req, rank, n_pes, True, False)
        _border_send_wait(r_send_t_req, r_send_t_req, rank, n_pes, True, False)

        # recv left
        if rank != 0:
            bodo.libs.distributed_api.wait(l_recv_req, True)
            bodo.libs.distributed_api.wait(l_recv_t_req, True)

            # values with start == 0 could potentially have left halo starts
            num_zero_starts = 0
            for i in range(0, N):
                if start[i] != 0:
                    break
                num_zero_starts += 1

            if num_zero_starts == 0:
                return output

            recv_starts = _get_var_recv_starts(
                on_arr, l_recv_t_buff, num_zero_starts, win
            )
            data = init_data()
            # setup (first element)
            for j in range(recv_starts[0], len(l_recv_t_buff)):
                data = add_obs(l_recv_buff[j], *data)
            if right_closed:
                data = add_obs(in_arr[0], *data)
            output[0] = calc_out(minp, *data)

            for i in range(1, num_zero_starts):
                s = recv_starts[i]
                e = end[i]

                # calculate deletes (can only happen in left recv buffer)
                for j in range(recv_starts[i - 1], s):
                    data = remove_obs(l_recv_buff[j], *data)

                # calculate adds (can only happen in local data)
                for j in range(end[i - 1], e):
                    data = add_obs(in_arr[j], *data)

                output[i] = calc_out(minp, *data)

    return output


@register_jitable(cache=True)
def _get_var_recv_starts(
    on_arr, l_recv_t_buff, num_zero_starts, win
):  # pragma: no cover
    recv_starts = np.zeros(num_zero_starts, np.int64)
    halo_size = len(l_recv_t_buff)
    index = cast_dt64_arr_to_int(on_arr)
    left_closed = False

    # handle first element
    start_bound = index[0] - win
    # left endpoint is closed
    if left_closed:
        start_bound -= 1
    recv_starts[0] = halo_size
    for j in range(0, halo_size):
        if l_recv_t_buff[j] > start_bound:
            recv_starts[0] = j
            break

    # rest of elements
    for i in range(1, num_zero_starts):
        start_bound = index[i] - win
        # left endpoint is closed
        if left_closed:
            start_bound -= 1
        recv_starts[i] = halo_size
        for j in range(recv_starts[i - 1], halo_size):
            if l_recv_t_buff[j] > start_bound:
                recv_starts[i] = j
                break

    return recv_starts


@register_jitable
def roll_var_linear_generic_seq(
    in_arr, on_arr, win, minp, start, end, init_data, add_obs, remove_obs, calc_out
):  # pragma: no cover
    #
    N = len(in_arr)
    output = np.empty(N, np.float64)

    data = init_data()

    # setup (first element)
    for j in range(start[0], end[0]):
        data = add_obs(in_arr[j], *data)

    output[0] = calc_out(minp, *data)

    for i in range(1, N):
        s = start[i]
        e = end[i]

        # calculate deletes
        for j in range(start[i - 1], s):
            data = remove_obs(in_arr[j], *data)

        # calculate adds
        for j in range(end[i - 1], e):
            data = add_obs(in_arr[j], *data)

        output[i] = calc_out(minp, *data)

    return output


def roll_variable_apply(
    in_arr, on_arr_dt, index_arr, win, minp, center, parallel, kernel_func, raw=True
):  # pragma: no cover
    pass


@overload(roll_variable_apply, no_unliteral=True)
def overload_roll_variable_apply(
    in_arr, on_arr_dt, index_arr, win, minp, center, parallel, kernel_func, raw=True
):
    assert_bodo_error(is_overload_constant_bool(raw))
    return roll_variable_apply_impl


dummy_use = numba.njit(lambda a: None)


def roll_variable_apply_impl(
    in_arr, on_arr_dt, index_arr, win, minp, center, parallel, kernel_func, raw=True
):  # pragma: no cover
    _validate_roll_var_args(minp, center)

    in_arr = prep_values(in_arr)
    win = offset_to_nanos(win)
    rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    on_arr = cast_dt64_arr_to_int(on_arr_dt)
    # replace index_arr=None argument (passed when index_arr is not needed) with dummy
    # array to avoid errors
    index_arr = fix_index_arr(index_arr)
    N = len(in_arr)
    # Pandas is right closed by default, TODO: extend to support arg
    left_closed = False
    right_closed = True

    if parallel:
        if _is_small_for_parallel_variable(on_arr, win):
            return _handle_small_data_variable_apply(
                in_arr, on_arr, index_arr, win, minp, rank, n_pes, kernel_func, raw
            )

        comm_data = _border_icomm_var(in_arr, on_arr, rank, n_pes, win)
        (
            l_recv_buff,
            l_recv_t_buff,
            r_send_req,
            r_send_t_req,
            l_recv_req,
            l_recv_t_req,
        ) = comm_data
        if raw == False:
            comm_data_idx = _border_icomm_var(index_arr, on_arr, rank, n_pes, win)
            (
                l_recv_buff_idx,
                l_recv_t_buff_idx,
                r_send_req_idx,
                r_send_t_req_idx,
                l_recv_req_idx,
                l_recv_t_req_idx,
            ) = comm_data_idx

    start, end = _build_indexer(on_arr, N, win, left_closed, right_closed)
    output = roll_variable_apply_seq(
        in_arr, on_arr, index_arr, win, minp, start, end, kernel_func, raw
    )

    if parallel:
        _border_send_wait(r_send_req, r_send_req, rank, n_pes, True, False)
        _border_send_wait(r_send_t_req, r_send_t_req, rank, n_pes, True, False)
        if raw == False:
            _border_send_wait(r_send_req_idx, r_send_req_idx, rank, n_pes, True, False)
            _border_send_wait(
                r_send_t_req_idx, r_send_t_req_idx, rank, n_pes, True, False
            )

        # recv left
        if rank != 0:
            bodo.libs.distributed_api.wait(l_recv_req, True)
            bodo.libs.distributed_api.wait(l_recv_t_req, True)
            if raw == False:
                bodo.libs.distributed_api.wait(l_recv_req_idx, True)
                bodo.libs.distributed_api.wait(l_recv_t_req_idx, True)

            # make sure unused buffer is not released before communication is done
            dummy_use(l_recv_t_buff_idx)

            # values with start == 0 could potentially have left halo starts
            num_zero_starts = 0
            for i in range(0, N):
                if start[i] != 0:
                    break
                num_zero_starts += 1

            if num_zero_starts == 0:
                return output

            recv_starts = _get_var_recv_starts(
                on_arr, l_recv_t_buff, num_zero_starts, win
            )
            recv_left_var_compute(
                output,
                in_arr,
                index_arr,
                num_zero_starts,
                recv_starts,
                l_recv_buff,
                l_recv_buff_idx,
                minp,
                kernel_func,
                raw,
            )

    return output


def recv_left_var_compute(
    output,
    in_arr,
    index_arr,
    num_zero_starts,
    recv_starts,
    l_recv_buff,
    l_recv_buff_idx,
    minp,
    kernel_func,
    raw,
):  # pragma: no cover
    pass


@overload(recv_left_var_compute)
def overload_recv_left_var_compute(
    output,
    in_arr,
    index_arr,
    num_zero_starts,
    recv_starts,
    l_recv_buff,
    l_recv_buff_idx,
    minp,
    kernel_func,
    raw,
):
    assert_bodo_error(is_overload_constant_bool(raw))
    if is_overload_true(raw):

        def impl(
            output,
            in_arr,
            index_arr,
            num_zero_starts,
            recv_starts,
            l_recv_buff,
            l_recv_buff_idx,
            minp,
            kernel_func,
            raw,
        ):  # pragma: no cover
            for i in range(0, num_zero_starts):
                halo_ind = recv_starts[i]
                sub_arr = np.concatenate((l_recv_buff[halo_ind:], in_arr[: i + 1]))
                if len(sub_arr) - np.isnan(sub_arr).sum() >= minp:
                    output[i] = kernel_func(sub_arr)
                else:
                    output[i] = np.nan

        return impl

    def impl_series(
        output,
        in_arr,
        index_arr,
        num_zero_starts,
        recv_starts,
        l_recv_buff,
        l_recv_buff_idx,
        minp,
        kernel_func,
        raw,
    ):  # pragma: no cover
        for i in range(0, num_zero_starts):
            halo_ind = recv_starts[i]
            sub_arr = np.concatenate((l_recv_buff[halo_ind:], in_arr[: i + 1]))
            sub_idx_arr = np.concatenate(
                (l_recv_buff_idx[halo_ind:], index_arr[: i + 1])
            )
            if len(sub_arr) - np.isnan(sub_arr).sum() >= minp:
                output[i] = kernel_func(pd.Series(sub_arr, sub_idx_arr))
            else:
                output[i] = np.nan

    return impl_series


def roll_variable_apply_seq(
    in_arr, on_arr, index_arr, win, minp, start, end, kernel_func, raw
):  # pragma: no cover
    pass


@overload(roll_variable_apply_seq)
def overload_roll_variable_apply_seq(
    in_arr, on_arr, index_arr, win, minp, start, end, kernel_func, raw
):
    assert_bodo_error(is_overload_constant_bool(raw))
    if is_overload_true(raw):
        return roll_variable_apply_seq_impl

    return roll_variable_apply_seq_impl_series


def roll_variable_apply_seq_impl(
    in_arr, on_arr, index_arr, win, minp, start, end, kernel_func, raw
):  # pragma: no cover
    N = len(in_arr)
    output = np.empty(N, dtype=np.float64)

    # TODO: handle count and minp
    for i in range(0, N):
        s = start[i]
        e = end[i]
        data = in_arr[s:e]
        if e - s - np.isnan(data).sum() >= minp:
            output[i] = kernel_func(data)
        else:
            output[i] = np.nan

    return output


# TODO(ehsan): avoid code duplication
def roll_variable_apply_seq_impl_series(
    in_arr, on_arr, index_arr, win, minp, start, end, kernel_func, raw
):  # pragma: no cover
    N = len(in_arr)
    output = np.empty(N, dtype=np.float64)

    # TODO: handle count and minp
    for i in range(0, N):
        s = start[i]
        e = end[i]
        data = in_arr[s:e]
        if e - s - np.isnan(data).sum() >= minp:
            output[i] = kernel_func(pd.Series(data, index_arr[s:e]))
        else:
            output[i] = np.nan

    return output


@register_jitable(cache=True)
def _build_indexer(on_arr, N, win, left_closed, right_closed):  # pragma: no cover
    index = cast_dt64_arr_to_int(on_arr)
    start = np.empty(N, np.int64)  # XXX pandas inits to -1 but doesn't seem required?
    end = np.empty(N, np.int64)
    start[0] = 0

    # right endpoint is closed
    if right_closed:
        end[0] = 1
    # right endpoint is open
    else:
        end[0] = 0

    # start is start of slice interval (including)
    # end is end of slice interval (not including)
    for i in range(1, N):
        end_bound = index[i]
        start_bound = index[i] - win

        # left endpoint is closed
        if left_closed:
            start_bound -= 1

        # advance the start bound until we are
        # within the constraint
        start[i] = i
        for j in range(start[i - 1], i):
            if index[j] > start_bound:
                start[i] = j
                break

        # end bound is previous end
        # or current index
        if index[end[i - 1]] <= end_bound:
            end[i] = i + 1
        else:
            end[i] = end[i - 1]

        # right endpoint is open
        if not right_closed:
            end[i] -= 1

    return start, end


# -------------------
# sum


@register_jitable
def init_data_sum():  # pragma: no cover
    return 0, 0.0


@register_jitable
def add_sum(val, nobs, sum_x):  # pragma: no cover
    if not np.isnan(val):
        nobs += 1
        sum_x += val
    return nobs, sum_x


@register_jitable
def remove_sum(val, nobs, sum_x):  # pragma: no cover
    if not np.isnan(val):
        nobs -= 1
        sum_x -= val
    return nobs, sum_x


@register_jitable
def calc_sum(minp, nobs, sum_x):  # pragma: no cover
    return sum_x if nobs >= minp else np.nan


# -------------------------------
# mean


@register_jitable
def init_data_mean():  # pragma: no cover
    return 0, 0.0, 0


@register_jitable
def add_mean(val, nobs, sum_x, neg_ct):  # pragma: no cover
    if not np.isnan(val):
        nobs += 1
        sum_x += val
        if val < 0:
            neg_ct += 1
    return nobs, sum_x, neg_ct


@register_jitable
def remove_mean(val, nobs, sum_x, neg_ct):  # pragma: no cover
    if not np.isnan(val):
        nobs -= 1
        sum_x -= val
        if val < 0:
            neg_ct -= 1
    return nobs, sum_x, neg_ct


@register_jitable
def calc_mean(minp, nobs, sum_x, neg_ct):  # pragma: no cover
    if nobs >= minp:
        result = sum_x / nobs
        if neg_ct == 0 and result < 0.0:
            # all positive
            result = 0
        elif neg_ct == nobs and result > 0.0:
            # all negative
            result = 0
    else:
        result = np.nan
    return result


# -------------------
# var

# TODO: combine add/remove similar to pandas?


@register_jitable
def init_data_var():  # pragma: no cover
    return 0, 0.0, 0.0


@register_jitable
def add_var(val, nobs, mean_x, ssqdm_x):  # pragma: no cover
    if not np.isnan(val):
        nobs += 1
        delta = val - mean_x
        mean_x += delta / nobs
        ssqdm_x += ((nobs - 1) * delta**2) / nobs
    return nobs, mean_x, ssqdm_x


@register_jitable
def remove_var(val, nobs, mean_x, ssqdm_x):  # pragma: no cover
    if not np.isnan(val):
        nobs -= 1
        if nobs != 0:
            delta = val - mean_x
            mean_x -= delta / nobs
            ssqdm_x -= ((nobs + 1) * delta**2) / nobs
        else:
            mean_x = 0.0
            ssqdm_x = 0.0
    return nobs, mean_x, ssqdm_x


@register_jitable
def calc_var(minp, nobs, mean_x, ssqdm_x):  # pragma: no cover
    ddof = 1.0  # TODO: make argument
    result = np.nan
    if nobs >= minp and nobs > ddof:
        # pathological case
        if nobs == 1:
            result = 0.0
        else:
            result = ssqdm_x / (nobs - ddof)
            if result < 0.0:
                result = 0.0

    return result


# --------------------------
# std


@register_jitable
def calc_std(minp, nobs, mean_x, ssqdm_x):  # pragma: no cover
    v = calc_var(minp, nobs, mean_x, ssqdm_x)
    return np.sqrt(v)


# -------------------
# count


@register_jitable
def init_data_count():  # pragma: no cover
    return (0.0,)


@register_jitable
def add_count(val, count_x):  # pragma: no cover
    if not np.isnan(val):
        count_x += 1.0
    return (count_x,)


@register_jitable
def remove_count(val, count_x):  # pragma: no cover
    if not np.isnan(val):
        count_x -= 1.0
    return (count_x,)


# XXX: pandas uses minp=0 for fixed window count but minp=1 for variable window


@register_jitable
def calc_count(minp, count_x):  # pragma: no cover
    return count_x


@register_jitable
def calc_count_var(minp, count_x):  # pragma: no cover
    return count_x if count_x >= minp else np.nan


# kernels for linear/streaming execution of rolling functions
linear_kernels = {
    "sum": (init_data_sum, add_sum, remove_sum, calc_sum),
    "mean": (init_data_mean, add_mean, remove_mean, calc_mean),
    "var": (init_data_var, add_var, remove_var, calc_var),
    "std": (init_data_var, add_var, remove_var, calc_std),
    "count": (init_data_count, add_count, remove_count, calc_count),
}


# shift -------------


# dummy
def shift(in_arr, shift, parallel, default_fill_value=None):  # pragma: no cover
    return


# using overload since njit bakes in Literal[bool](False) for parallel
@overload(shift, jit_options={"cache": True})
def shift_overload(in_arr, shift, parallel, default_fill_value=None):
    # TODO: Removing this check passes our internal shift tests.
    # We should remove this check if possible so this implementation
    # always returns a function.
    if not isinstance(parallel, types.Literal):
        return shift_impl


def shift_impl(in_arr, shift, parallel, default_fill_value=None):  # pragma: no cover
    N = len(in_arr)
    # fallback to regular string array if dictionary-encoded array
    # TODO(ehsan): support dictionary-encoded arrays directly
    in_arr = decode_if_dict_array(in_arr)
    output = alloc_shift(N, in_arr, (-1,), fill_value=default_fill_value)
    send_right = shift > 0
    send_left = shift <= 0
    is_parallel_str = False
    if parallel:
        rank = bodo.libs.distributed_api.get_rank()
        n_pes = bodo.libs.distributed_api.get_size()
        halo_size = np.int32(abs(shift))
        if _is_small_for_parallel(N, halo_size):
            return _handle_small_data_shift(
                in_arr, shift, rank, n_pes, default_fill_value
            )

        comm_data = _border_icomm(in_arr, rank, n_pes, halo_size, send_right, send_left)
        (
            l_recv_buff,
            r_recv_buff,
            l_send_req,
            r_send_req,
            l_recv_req,
            r_recv_req,
        ) = comm_data

        # update start of output array (from left recv buff) early for string arrays
        # since they are immutable and should be written in order
        if send_right and is_str_binary_array(in_arr):
            is_parallel_str = True
            shift_left_recv(
                r_send_req,
                l_send_req,
                rank,
                n_pes,
                halo_size,
                l_recv_req,
                l_recv_buff,
                output,
            )

    shift_seq(in_arr, shift, output, is_parallel_str, default_fill_value)

    if parallel:
        if send_right:
            if not is_str_binary_array(in_arr):
                shift_left_recv(
                    r_send_req,
                    l_send_req,
                    rank,
                    n_pes,
                    halo_size,
                    l_recv_req,
                    l_recv_buff,
                    output,
                )

        else:
            _border_send_wait(r_send_req, l_send_req, rank, n_pes, False, True)

            # recv right
            if rank != n_pes - 1:
                bodo.libs.distributed_api.wait(r_recv_req, True)

                for i in range(0, halo_size):
                    if bodo.libs.array_kernels.isna(r_recv_buff, i):
                        bodo.libs.array_kernels.setna(output, N - halo_size + i)
                        continue
                    output[N - halo_size + i] = r_recv_buff[i]

    return output


@register_jitable(cache=True)
def shift_seq(
    in_arr, shift, output, is_parallel_str=False, default_fill_value=None
):  # pragma: no cover
    N = len(in_arr)
    # maximum shift size is N
    sign_shift = 1 if shift > 0 else -1
    shift = sign_shift * min(abs(shift), N)
    # set border values to NA. We skip this for parallel string arrays because
    # their values were already set in order, except for rank 0.
    # We never need to do this for rank != 0 because if shift > N
    # another code path is chosen.
    if shift > 0 and (not is_parallel_str or bodo.get_rank() == 0):
        if default_fill_value is None:
            bodo.libs.array_kernels.setna_slice(output, slice(None, shift))
        else:
            for i in range(shift):
                output[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(
                    default_fill_value
                )

    # range is shift..N for positive shift, 0..N+shift for negative shift
    start = max(shift, 0)
    end = min(N, N + shift)

    for i in range(start, end):
        if bodo.libs.array_kernels.isna(in_arr, i - shift):
            bodo.libs.array_kernels.setna(output, i)
            continue
        output[i] = in_arr[i - shift]

    # NOTE: updating end of array later since string arrays require in order setitem
    if shift < 0:
        if default_fill_value is None:
            bodo.libs.array_kernels.setna_slice(output, slice(shift, None))
        else:
            for i in range(end, N):
                output[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(
                    default_fill_value
                )

    return output


@register_jitable
def shift_left_recv(
    r_send_req, l_send_req, rank, n_pes, halo_size, l_recv_req, l_recv_buff, output
):  # pragma: no cover
    """wait for send/recv comm calls and update output using left recv buff for shift()"""
    _border_send_wait(r_send_req, l_send_req, rank, n_pes, True, False)

    # recv left
    if rank != 0:
        bodo.libs.distributed_api.wait(l_recv_req, True)

        for i in range(0, halo_size):
            if bodo.libs.array_kernels.isna(l_recv_buff, i):
                bodo.libs.array_kernels.setna(output, i)
                continue
            output[i] = l_recv_buff[i]


def is_str_binary_array(arr):  # pragma: no cover
    return False


@overload(is_str_binary_array)
def overload_is_str_binary_array(arr):
    """return True if 'arr' is a string or binary array"""
    if arr in [bodo.types.string_array_type, bodo.types.binary_array_type]:
        return lambda arr: True  # pragma: no cover

    return lambda arr: False  # pragma: no cover


def is_supported_shift_array_type(arr_type):
    """return True if array type is supported for shift() operation"""
    return (
        (
            isinstance(arr_type, types.Array)
            and (
                isinstance(arr_type.dtype, types.Number)
                or arr_type.dtype in [bodo.types.datetime64ns, bodo.types.timedelta64ns]
            )
        )
        or isinstance(
            arr_type,
            (
                bodo.types.IntegerArrayType,
                bodo.types.FloatingArrayType,
                bodo.types.DecimalArrayType,
                bodo.types.DatetimeArrayType,
                bodo.types.TimeArrayType,
            ),
        )
        or arr_type
        in (
            bodo.types.boolean_array_type,
            bodo.types.datetime_date_array_type,
            bodo.types.string_array_type,
            bodo.types.binary_array_type,
            bodo.types.dict_str_arr_type,
        )
    )


# pct_change -------------


# dummy
def pct_change():  # pragma: no cover
    return


# using overload since njit bakes in Literal[bool](False) for parallel
@overload(pct_change, jit_options={"cache": True})
def pct_change_overload(in_arr, shift, parallel):
    if not isinstance(parallel, types.Literal):
        return pct_change_impl


def pct_change_impl(in_arr, shift, parallel):  # pragma: no cover
    N = len(in_arr)
    send_right = shift > 0
    send_left = shift <= 0
    if parallel:
        rank = bodo.libs.distributed_api.get_rank()
        n_pes = bodo.libs.distributed_api.get_size()
        halo_size = np.int32(abs(shift))
        if _is_small_for_parallel(N, halo_size):
            return _handle_small_data_pct_change(in_arr, shift, rank, n_pes)

        comm_data = _border_icomm(in_arr, rank, n_pes, halo_size, send_right, send_left)
        (
            l_recv_buff,
            r_recv_buff,
            l_send_req,
            r_send_req,
            l_recv_req,
            r_recv_req,
        ) = comm_data

    output = pct_change_seq(in_arr, shift)

    if parallel:
        if send_right:
            _border_send_wait(r_send_req, l_send_req, rank, n_pes, True, False)

            # recv left
            if rank != 0:
                bodo.libs.distributed_api.wait(l_recv_req, True)

                for i in range(0, halo_size):
                    prev = l_recv_buff[i]
                    output[i] = (in_arr[i] - prev) / prev
        else:
            _border_send_wait(r_send_req, l_send_req, rank, n_pes, False, True)

            # recv right
            if rank != n_pes - 1:
                bodo.libs.distributed_api.wait(r_recv_req, True)

                for i in range(0, halo_size):
                    prev = r_recv_buff[i]
                    output[N - halo_size + i] = (
                        in_arr[N - halo_size + i] - prev
                    ) / prev

    return output


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_first_non_na(arr):
    """get first non-NA value of numeric array."""
    # just return 0 for non-floats
    if isinstance(arr.dtype, (types.Integer, types.Boolean)):
        zero = arr.dtype(0)
        return lambda arr: zero if len(arr) == 0 else arr[0]  # pragma: no cover

    assert isinstance(arr.dtype, types.Float)
    # TODO: int array

    na_val = np.nan
    if arr.dtype == types.float32:
        na_val = np.float32("nan")

    def impl(arr):  # pragma: no cover
        for i in range(len(arr)):
            if not bodo.libs.array_kernels.isna(arr, i):
                return arr[i]

        return na_val

    return impl


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_last_non_na(arr):
    """get last non-NA value of numeric array."""
    # just return 0 for non-floats
    if isinstance(arr.dtype, (types.Integer, types.Boolean)):
        zero = arr.dtype(0)
        return lambda arr: zero if len(arr) == 0 else arr[-1]  # pragma: no cover

    assert isinstance(arr.dtype, types.Float)
    # TODO: int array

    na_val = np.nan
    if arr.dtype == types.float32:
        na_val = np.float32("nan")

    def impl(arr):  # pragma: no cover
        l = len(arr)
        for i in range(len(arr)):
            ind = l - i - 1
            if not bodo.libs.array_kernels.isna(arr, ind):
                return arr[ind]

        return na_val

    return impl


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_one_from_arr_dtype(arr):
    one = arr.dtype(1)
    return lambda arr: one  # pragma: no cover


@register_jitable(cache=True)
def pct_change_seq(in_arr, shift):  # pragma: no cover
    # TODO: parallel 'pad' fill
    N = len(in_arr)
    output = alloc_pct_change(N, in_arr)
    # maximum shift size is N
    sign_shift = 1 if shift > 0 else -1
    shift = sign_shift * min(abs(shift), N)
    # set border values to NA
    if shift > 0:
        bodo.libs.array_kernels.setna_slice(output, slice(None, shift))
    else:
        bodo.libs.array_kernels.setna_slice(output, slice(shift, None))

    # using 'pad' method for handling NAs, TODO: support bfill
    if shift > 0:
        fill_prev = get_first_non_na(in_arr[:shift])
        fill = get_last_non_na(in_arr[:shift])
    else:
        fill_prev = get_last_non_na(in_arr[:-shift])
        fill = get_first_non_na(in_arr[:-shift])

    one = get_one_from_arr_dtype(output)

    # range is shift..N for positive shift, 0..N+shift for negative shift
    start = max(shift, 0)
    end = min(N, N + shift)

    for i in range(start, end):
        prev = in_arr[i - shift]
        # TODO: support non-float output (e.g. timedelta64?)
        if np.isnan(prev):
            prev = fill_prev
        else:
            fill_prev = prev
        val = in_arr[i]
        if np.isnan(val):
            val = fill
        else:
            fill = val
        output[i] = val / prev - one

    return output


# communication calls -----------


@register_jitable(cache=True)
def _border_icomm(
    in_arr, rank, n_pes, halo_size, send_right=True, send_left=False
):  # pragma: no cover
    """post isend/irecv for halo data (fixed window case)"""
    comm_tag = np.int32(comm_border_tag)
    l_recv_buff = bodo.utils.utils.alloc_type(halo_size, in_arr, (-1,))
    r_recv_buff = bodo.utils.utils.alloc_type(halo_size, in_arr, (-1,))

    # send right
    if send_right and rank != n_pes - 1:
        r_send_req = bodo.libs.distributed_api.isend(
            in_arr[-halo_size:], halo_size, np.int32(rank + 1), comm_tag, True
        )
    # recv left
    if send_right and rank != 0:
        l_recv_req = bodo.libs.distributed_api.irecv(
            l_recv_buff, halo_size, np.int32(rank - 1), comm_tag, True
        )
    # send_left cases
    # send left
    if send_left and rank != 0:
        l_send_req = bodo.libs.distributed_api.isend(
            in_arr[:halo_size], halo_size, np.int32(rank - 1), comm_tag, True
        )
    # recv right
    if send_left and rank != n_pes - 1:
        r_recv_req = bodo.libs.distributed_api.irecv(
            r_recv_buff, halo_size, np.int32(rank + 1), comm_tag, True
        )

    return l_recv_buff, r_recv_buff, l_send_req, r_send_req, l_recv_req, r_recv_req


@register_jitable(cache=True)
def _border_icomm_var(in_arr, on_arr, rank, n_pes, win_size):  # pragma: no cover
    comm_tag = np.int32(comm_border_tag)
    # find halo size from time array
    N = len(on_arr)
    halo_size = N
    end = on_arr[-1]
    for j in range(-2, -N, -1):
        t = on_arr[j]
        if end - t >= win_size:
            halo_size = -j
            break

    # send right
    if rank != n_pes - 1:
        bodo.libs.distributed_api.send(halo_size, np.int32(rank + 1), comm_tag)
        r_send_req = bodo.libs.distributed_api.isend(
            in_arr[-halo_size:], np.int32(halo_size), np.int32(rank + 1), comm_tag, True
        )
        r_send_t_req = bodo.libs.distributed_api.isend(
            on_arr[-halo_size:], np.int32(halo_size), np.int32(rank + 1), comm_tag, True
        )
    # recv left
    if rank != 0:
        halo_size = bodo.libs.distributed_api.recv(
            np.int64, np.int32(rank - 1), comm_tag
        )
        l_recv_buff = bodo.utils.utils.alloc_type(halo_size, in_arr)
        l_recv_req = bodo.libs.distributed_api.irecv(
            l_recv_buff, np.int32(halo_size), np.int32(rank - 1), comm_tag, True
        )
        l_recv_t_buff = np.empty(halo_size, np.int64)
        l_recv_t_req = bodo.libs.distributed_api.irecv(
            l_recv_t_buff, np.int32(halo_size), np.int32(rank - 1), comm_tag, True
        )

    return (
        l_recv_buff,
        l_recv_t_buff,
        r_send_req,
        r_send_t_req,
        l_recv_req,
        l_recv_t_req,
    )


@register_jitable
def _border_send_wait(
    r_send_req, l_send_req, rank, n_pes, right, left
):  # pragma: no cover
    # wait on send right
    if right and rank != n_pes - 1:
        bodo.libs.distributed_api.wait(r_send_req, True)
    # wait on send left
    if left and rank != 0:
        bodo.libs.distributed_api.wait(l_send_req, True)


@register_jitable
def _is_small_for_parallel(N, halo_size):  # pragma: no cover
    # gather data on one processor and compute sequentially if data of any
    # processor is too small for halo size
    # TODO: handle 1D_Var or other cases where data is actually large but
    # highly imbalanced
    # TODO: avoid reduce for obvious cases like no center and large 1D_Block
    # using 2*halo_size+1 to accomodate center cases with data on more than
    # 2 processor
    num_small = bodo.libs.distributed_api.dist_reduce(
        int(N <= 2 * halo_size + 1), np.int32(Reduce_Type.Sum.value)
    )
    return num_small != 0


# TODO: refactor small data functions
@register_jitable
def _handle_small_data(
    in_arr, win, minp, center, rank, n_pes, init_data, add_obs, remove_obs, calc_out
):  # pragma: no cover
    """Gather data and run rolling window computation (fixed window case) on rank 0,
    then broadcast the result.
    This is used when input data is too small compared to window size for efficient
    parallelism.
    """
    N = len(in_arr)
    all_N = bodo.libs.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value)
    )
    all_in_arr = bodo.libs.distributed_api.gatherv(in_arr)
    if rank == 0:
        all_out, _ = roll_fixed_linear_generic_seq(
            all_in_arr, win, minp, center, init_data, add_obs, remove_obs, calc_out
        )
    else:
        all_out = np.empty(all_N, np.float64)
    bodo.libs.distributed_api.bcast_preallocated(all_out)
    # 1D_Var chunk sizes can be variable, TODO: use 1D flag to avoid exscan
    start = bodo.libs.distributed_api.dist_exscan(N, np.int32(Reduce_Type.Sum.value))
    end = start + N
    # start = bodo.libs.distributed_api.get_start(all_N, n_pes, rank)
    # end = bodo.libs.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]


@register_jitable
def _handle_small_data_apply(
    in_arr, index_arr, win, minp, center, rank, n_pes, kernel_func, raw=True
):  # pragma: no cover
    """Gather data and run rolling window computation (fixed window apply case) on rank
    0, then broadcast the result.
    This is used when input data is too small compared to window size for efficient
    parallelism.
    """
    N = len(in_arr)
    all_N = bodo.libs.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value)
    )
    all_in_arr = bodo.libs.distributed_api.gatherv(in_arr)
    all_index_arr = bodo.libs.distributed_api.gatherv(index_arr)
    if rank == 0:
        all_out = roll_fixed_apply_seq(
            all_in_arr, all_index_arr, win, minp, center, kernel_func, raw
        )
    else:
        all_out = np.empty(all_N, np.float64)
    bodo.libs.distributed_api.bcast_preallocated(all_out)
    # 1D_Var chunk sizes can be variable, TODO: use 1D flag to avoid exscan
    start = bodo.libs.distributed_api.dist_exscan(N, np.int32(Reduce_Type.Sum.value))
    end = start + N
    # start = bodo.libs.distributed_api.get_start(all_N, n_pes, rank)
    # end = bodo.libs.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]


def bcast_n_chars_if_str_binary_arr(arr):
    pass


@overload(bcast_n_chars_if_str_binary_arr)
def overload_bcast_n_chars_if_str_binary_arr(arr):
    """broadcast number of characters if 'arr' is a string or binary array"""
    if arr in [bodo.types.binary_array_type, bodo.types.string_array_type]:

        def impl(arr):  # pragma: no cover
            return bodo.libs.distributed_api.bcast_scalar(
                np.int64(bodo.libs.str_arr_ext.num_total_chars(arr))
            )

        return impl

    return lambda arr: -1  # pragma: no cover


@register_jitable
def _handle_small_data_shift(
    in_arr, shift, rank, n_pes, default_fill_value
):  # pragma: no cover
    """Gather data and run shift computation on rank 0,
    then broadcast the result.
    This is used when input data is too small compared to window size for efficient
    parallelism.
    """
    N = len(in_arr)
    all_N = bodo.libs.distributed_api.dist_reduce(
        len(in_arr), np.int32(Reduce_Type.Sum.value)
    )
    all_in_arr = bodo.libs.distributed_api.gatherv(in_arr)
    if rank == 0:
        all_out = alloc_shift(
            len(all_in_arr), all_in_arr, (-1,), fill_value=default_fill_value
        )
        shift_seq(all_in_arr, shift, all_out, default_fill_value=default_fill_value)
        n_chars = bcast_n_chars_if_str_binary_arr(all_out)
    else:
        n_chars = bcast_n_chars_if_str_binary_arr(in_arr)
        all_out = alloc_shift(all_N, in_arr, (n_chars,), fill_value=default_fill_value)

    bodo.libs.distributed_api.bcast_preallocated(all_out)
    # 1D_Var chunk sizes can be variable, TODO: use 1D flag to avoid exscan
    start = bodo.libs.distributed_api.dist_exscan(N, np.int32(Reduce_Type.Sum.value))
    end = start + N
    # start = bodo.libs.distributed_api.get_start(all_N, n_pes, rank)
    # end = bodo.libs.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]


@register_jitable
def _handle_small_data_pct_change(in_arr, shift, rank, n_pes):  # pragma: no cover
    """Gather data and run pct_change computation on rank 0,
    then broadcast the result.
    This is used when input data is too small compared to window size for efficient
    parallelism.
    """
    N = len(in_arr)
    all_N = bodo.libs.distributed_api.dist_reduce(N, np.int32(Reduce_Type.Sum.value))
    all_in_arr = bodo.libs.distributed_api.gatherv(in_arr)
    if rank == 0:
        all_out = pct_change_seq(all_in_arr, shift)
    else:
        all_out = alloc_pct_change(all_N, in_arr)
    bodo.libs.distributed_api.bcast_preallocated(all_out)
    # 1D_Var chunk sizes can be variable, TODO: use 1D flag to avoid exscan
    start = bodo.libs.distributed_api.dist_exscan(N, np.int32(Reduce_Type.Sum.value))
    end = start + N
    # start = bodo.libs.distributed_api.get_start(all_N, n_pes, rank)
    # end = bodo.libs.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]


def cast_dt64_arr_to_int(arr):  # pragma: no cover
    return arr


@infer_global(cast_dt64_arr_to_int)
class DtArrToIntType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        assert (
            args[0] == types.Array(types.NPDatetime("ns"), 1, "C")
            or args[0] == types.Array(types.int64, 1, "C")
            or args[0] == bodo.types.DatetimeArrayType(None)
        )
        return signature(types.Array(types.int64, 1, "C"), *args)


@lower_builtin(cast_dt64_arr_to_int, types.Array(types.NPDatetime("ns"), 1, "C"))
@lower_builtin(cast_dt64_arr_to_int, types.Array(types.int64, 1, "C"))
def lower_cast_dt64_arr_to_int(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


@lower_builtin(cast_dt64_arr_to_int, bodo.types.DatetimeArrayType(None))
def lower_cast_datetime_arr_to_int(context, builder, sig, args):
    datetime_struct = cgutils.create_struct_proxy(sig.args[0])(
        context, builder, value=args[0]
    )
    return impl_ret_borrowed(context, builder, sig.return_type, datetime_struct.data)


# ----------------------------------------
# variable window comm routines


@register_jitable
def _is_small_for_parallel_variable(on_arr, win_size):  # pragma: no cover
    # assume small if current processor's whole range is smaller than win_size
    if len(on_arr) < 2:
        is_small = 1
    else:
        start = on_arr[0]
        end = on_arr[-1]
        pe_range = end - start
        is_small = int(pe_range <= win_size)
    num_small = bodo.libs.distributed_api.dist_reduce(
        is_small, np.int32(Reduce_Type.Sum.value)
    )
    return num_small != 0


@register_jitable
def _handle_small_data_variable(
    in_arr, on_arr, win, minp, rank, n_pes, init_data, add_obs, remove_obs, calc_out
):  # pragma: no cover
    """Gather data and run rolling window computation (variable window case) on rank 0,
    then broadcast the result.
    This is used when input data is too small compared to window size for efficient
    parallelism.
    """
    N = len(in_arr)
    all_N = bodo.libs.distributed_api.dist_reduce(N, np.int32(Reduce_Type.Sum.value))
    all_in_arr = bodo.libs.distributed_api.gatherv(in_arr)
    all_on_arr = bodo.libs.distributed_api.gatherv(on_arr)
    if rank == 0:
        start, end = _build_indexer(all_on_arr, all_N, win, False, True)
        all_out = roll_var_linear_generic_seq(
            all_in_arr,
            all_on_arr,
            win,
            minp,
            start,
            end,
            init_data,
            add_obs,
            remove_obs,
            calc_out,
        )
    else:
        all_out = np.empty(all_N, np.float64)
    bodo.libs.distributed_api.bcast_preallocated(all_out)
    # 1D_Var chunk sizes can be variable, TODO: use 1D flag to avoid exscan
    start = bodo.libs.distributed_api.dist_exscan(N, np.int32(Reduce_Type.Sum.value))
    end = start + N
    # start = bodo.libs.distributed_api.get_start(all_N, n_pes, rank)
    # end = bodo.libs.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]


@register_jitable
def _handle_small_data_variable_apply(
    in_arr, on_arr, index_arr, win, minp, rank, n_pes, kernel_func, raw
):  # pragma: no cover
    """Gather data and run rolling window computation (variable window apply case) on,
    rank 0 then broadcast the result.
    This is used when input data is too small compared to window size for efficient
    parallelism.
    """
    N = len(in_arr)
    all_N = bodo.libs.distributed_api.dist_reduce(N, np.int32(Reduce_Type.Sum.value))
    all_in_arr = bodo.libs.distributed_api.gatherv(in_arr)
    all_on_arr = bodo.libs.distributed_api.gatherv(on_arr)
    all_index_arr = bodo.libs.distributed_api.gatherv(index_arr)
    if rank == 0:
        start, end = _build_indexer(all_on_arr, all_N, win, False, True)
        all_out = roll_variable_apply_seq(
            all_in_arr,
            all_on_arr,
            all_index_arr,
            win,
            minp,
            start,
            end,
            kernel_func,
            raw,
        )
    else:
        all_out = np.empty(all_N, np.float64)
    bodo.libs.distributed_api.bcast_preallocated(all_out)
    # 1D_Var chunk sizes can be variable, TODO: use 1D flag to avoid exscan
    start = bodo.libs.distributed_api.dist_exscan(N, np.int32(Reduce_Type.Sum.value))
    end = start + N
    # start = bodo.libs.distributed_api.get_start(all_N, n_pes, rank)
    # end = bodo.libs.distributed_api.get_end(all_N, n_pes, rank)
    return all_out[start:end]


@register_jitable(cache=True)
def _dropna(arr):  # pragma: no cover
    old_len = len(arr)
    new_len = old_len - np.isnan(arr).sum()
    A = np.empty(new_len, arr.dtype)
    curr_ind = 0
    for i in range(old_len):
        val = arr[i]
        if not np.isnan(val):
            A[curr_ind] = val
            curr_ind += 1

    return A


def alloc_shift(n, A, s=None, fill_value=None):  # pragma: no cover
    return np.empty(n, A.dtype)


@overload(alloc_shift, no_unliteral=True)
def alloc_shift_overload(n, A, s=None, fill_value=None):
    """allocate output array for shift(). It is the same type as input, except for
    non-nullable int case which requires float (to store nulls).
    """

    # non-Numpy case is same as input
    if not isinstance(A, types.Array):
        return lambda n, A, s=None, fill_value=None: bodo.utils.utils.alloc_type(
            n, A, s
        )  # pragma: no cover

    # output of non-nullable int is float64 to be able to store nulls,
    # unless a integer fill value is provided
    if isinstance(A.dtype, types.Integer) and not isinstance(fill_value, types.Integer):
        return lambda n, A, s=None, fill_value=None: np.empty(
            n, np.float64
        )  # pragma: no cover

    return lambda n, A, s=None, fill_value=None: np.empty(
        n, A.dtype
    )  # pragma: no cover


def alloc_pct_change(n, A):  # pragma: no cover
    return np.empty(n, A.dtype)


@overload(alloc_pct_change, no_unliteral=True)
def alloc_pct_change_overload(n, A):
    """allocate output array for pct_change(). The output is float for int input."""

    # output of Numpy int is float to set NAs
    if isinstance(A.dtype, types.Integer):
        return lambda n, A: np.empty(n, np.float64)  # pragma: no cover

    return lambda n, A: bodo.utils.utils.alloc_type(n, A, (-1,))  # pragma: no cover


def prep_values(A):  # pragma: no cover
    return A.astype("float64")


@overload(prep_values, no_unliteral=True)
def prep_values_overload(A):
    # Pandas converts rolling input to float64
    # https://github.com/pandas-dev/pandas/blob/6d0dab4c0914031517be4a3d1aff999e25cc2649/pandas/core/window/rolling.py#L265

    # NOTE: A.astype("float64", copy=False) doesn't work in Numba (TODO: fix)
    if A == types.Array(types.float64, 1, "C"):
        return lambda A: A  # pragma: no cover

    return lambda A: A.astype(np.float64)  # pragma: no cover


@register_jitable
def _validate_roll_fixed_args(win, minp):  # pragma: no cover
    """error checking for arguments to rolling with fixed window"""
    if win < 0:
        raise ValueError("window must be non-negative")

    if minp < 0:
        raise ValueError("min_periods must be >= 0")

    if minp > win:
        # add minp/win values to error when possible in Numba (string should be const
        # currently)
        raise ValueError("min_periods must be <= window")


@register_jitable
def _validate_roll_var_args(minp, center):  # pragma: no cover
    """error checking for arguments to rolling with offset window"""
    if minp < 0:
        raise ValueError("min_periods must be >= 0")

    # TODO(ehsan): make sure on_arr_dt is monotonic

    if center:
        raise NotImplementedError(
            "rolling: center is not implemented for "
            "datetimelike and offset based windows"
        )
