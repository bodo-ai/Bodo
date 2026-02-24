"""
Implements wrappers to call the C++ BodoSQL array kernels for LEAD/LAG.
"""

import textwrap

import llvmlite.binding as ll
import numba
from numba.core import types
from numba.extending import intrinsic

import bodo
from bodo.libs import lead_lag
from bodo.libs.array import array_to_info, delete_info, info_to_array
from bodo.utils.typing import (
    dtype_to_array_type,
    is_nullable,
    raise_bodo_error,
    to_nullable_type,
)
from bodosql.kernels.array_kernel_utils import (
    is_valid_binary_arg,
    is_valid_string_arg,
)

# lead_lag_seq

# load lead_lag_seq_py_entry function from C++
ll.add_symbol(
    "lead_lag_seq_py_entry",
    lead_lag.lead_lag_seq_py_entry,
)

_lead_lag_seq_py_entry = types.ExternalFunction(
    "lead_lag_seq_py_entry",
    bodo.libs.array.array_info_type(
        bodo.libs.array.array_info_type,
        types.int64,
        types.voidptr,
        types.int64,
        types.boolean,
    ),
)


@intrinsic
def ptr_to_voidptr(typingctx, val):
    """Convert type of input to voidptr (just type change with no runtime value change)"""

    def codegen(context, builder, signature, args):
        return args[0]

    return types.voidptr(val), codegen


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def lead_lag_seq(in_col, shift_amt, default_fill_val=None, ignore_nulls=False):
    """
    The function will effectively shift all values of the array by
    shift_amt, whether it is negative or positive. In the process, any values
    that correspond to out of bounds indices in the input are set to the
    default_fill_val, or null if it is not present

    Args:
        in_col (pd.series/np.ndarray): Column
        shift_amt (int): amount to shift by--can be positive or negative.
        default_fill_val (any): Value to fill out-of-bounds values with. Default None, corresponding to null.
            Type must reasonably match array dtype.
    Returns:
        pd.series: a single column with the lead/lag operation performed on it.
    """

    # Create substitutions dictionary for func_text
    ctx = {}
    if isinstance(in_col, bodo.types.SeriesType):  # pragma: no cover
        in_col = in_col.data
        ctx["array_conv"] = "in_col = bodo.utils.conversion.coerce_to_array(in_col)"
    else:
        ctx["array_conv"] = ""

    if (
        in_col == bodo.types.dict_str_arr_type
        or in_col == bodo.types.timedelta_array_type
    ):
        result_type = in_col
    else:
        result_type = to_nullable_type(dtype_to_array_type(in_col.dtype))

    # Make sure numpy input types returns nullable
    if is_nullable(in_col):
        ctx["return_in_col_nullable"] = "return in_col"
    else:
        ctx["return_in_col_nullable"] = (
            "return bodo.utils.conversion.coerce_to_array(in_col, use_nullable_array=True)"
        )

    # Handle necessary conversions for certain input types.
    ctx["fill_val_post_call"] = ""
    if isinstance(default_fill_val, types.NoneType):  # pragma: no cover
        ctx["fill_val"] = "fill_val = 0"
    elif is_valid_string_arg(default_fill_val):  # pragma: no cover
        ctx["fill_val"] = (
            "(fill_val, fill_val_len) = bodo.libs.str_ext.unicode_to_utf8_and_len(default_fill_val)"
        )
    elif is_valid_binary_arg(default_fill_val):
        ctx["fill_val"] = (
            "(fill_val, fill_val_len) = (ptr_to_voidptr(default_fill_val._data), len(default_fill_val))"
        )
    elif isinstance(default_fill_val, types.StringLiteral):
        raise_bodo_error("Lead/lag does not support StringLiteral.")
    else:
        # create an array with the fill value to be able to pass a pointer to C++
        data_ptr = (
            "default_arr._data.ctypes"
            if isinstance(
                in_col,
                (
                    bodo.types.DecimalArrayType,
                    bodo.types.TimeArrayType,
                ),
            )
            or in_col == bodo.types.datetime_date_array_type
            or (
                isinstance(in_col, bodo.types.DatetimeArrayType)
                and in_col.tz is not None
            )
            else "default_arr.ctypes"
        )
        ctx["fill_val"] = (
            f"default_arr = bodo.utils.conversion.coerce_scalar_to_array(default_fill_val, 1, in_col); fill_val = {data_ptr}"
        )
        ctx["fill_val_post_call"] = "dummy_use(default_arr)"
    func_text = textwrap.dedent(
        f"""
    def _lead_lag_seq_impl(in_col, shift_amt, default_fill_val=None, ignore_nulls=False):
        fill_val_len = 0
        {ctx["array_conv"]}

        if shift_amt == 0:
            {ctx["return_in_col_nullable"]}

        in_col_info = array_to_info(in_col)
        {ctx["fill_val"]}
        result_info = _lead_lag_seq_py_entry(
            in_col_info,
            shift_amt,
            fill_val,
            fill_val_len,
            ignore_nulls,
        )
        {ctx["fill_val_post_call"]}
        result = info_to_array(
            result_info,
            result_type,
        )
        delete_info(result_info)
        return result
    """
    )
    global_vars = {
        "result_type": result_type,
        "bodo": bodo,
        "array_to_info": array_to_info,
        "info_to_array": info_to_array,
        "delete_info": delete_info,
        "_lead_lag_seq_py_entry": _lead_lag_seq_py_entry,
        "dummy_use": numba.njit(lambda a: None),
        "ptr_to_voidptr": ptr_to_voidptr,
    }
    local_vars = {}
    exec(func_text, global_vars, local_vars)
    return local_vars["_lead_lag_seq_impl"]
