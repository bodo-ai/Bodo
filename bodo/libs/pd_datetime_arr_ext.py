# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""DatetimeArray extension for Pandas DatetimeArray with timezone support."""

import operator

import numba
import numpy as np
import pandas as pd
import pytz
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import lower_constant
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_jitable,
    register_model,
    typeof_impl,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.utils.conversion import ensure_contig_if_np
from bodo.utils.typing import (
    BodoArrayIterator,
    BodoError,
    get_literal_value,
    is_list_like_index_type,
    is_overload_constant_int,
    is_overload_constant_str,
)


class PandasDatetimeTZDtype(types.Type):
    """Data type for datetime timezone"""

    def __init__(self, tz):
        if isinstance(tz, (pytz._FixedOffset, pytz.tzinfo.BaseTzInfo)):
            tz = get_pytz_type_info(tz)
        if not isinstance(tz, (int, str)):
            raise BodoError(
                "Timezone must be either a valid pytz type with a zone or a fixed offset"
            )
        self.tz = tz
        super(PandasDatetimeTZDtype, self).__init__(name=f"PandasDatetimeTZDtype[{tz}]")


register_model(PandasDatetimeTZDtype)(models.OpaqueModel)


@lower_constant(PandasDatetimeTZDtype)
def lower_constant_pd_datetime_tz_dtype(context, builder, typ, pyval):
    return context.get_dummy_value()


@box(PandasDatetimeTZDtype)
def box_pd_datetime_tzdtype(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    # Create the timezone type.
    unit_str = c.context.get_constant_generic(c.builder, types.unicode_type, "ns")
    # No need to incref because unit_str is a constant
    unit_str_obj = c.pyapi.from_native_value(
        types.unicode_type, unit_str, c.env_manager
    )
    if isinstance(typ.tz, str):
        tz_str = c.context.get_constant_generic(c.builder, types.unicode_type, typ.tz)
        # No need to incref because tz_str is a constant
        tz_arg_obj = c.pyapi.from_native_value(
            types.unicode_type, tz_str, c.env_manager
        )
    else:
        # We store ns, but the Fixed offset constructor takes minutes.
        offset = nanoseconds_to_offset(typ.tz)
        tz_arg_obj = c.pyapi.unserialize(c.pyapi.serialize_object(offset))

    res = c.pyapi.call_method(
        pd_class_obj, "DatetimeTZDtype", (unit_str_obj, tz_arg_obj)
    )
    c.pyapi.decref(unit_str_obj)
    c.pyapi.decref(tz_arg_obj)
    c.pyapi.decref(pd_class_obj)
    # decref() should be called on native value
    # see https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/boxing.py#L389
    c.context.nrt.decref(c.builder, typ, val)
    return res


@unbox(PandasDatetimeTZDtype)
def unbox_pd_datetime_tzdtype(typ, val, c):
    return NativeValue(c.context.get_dummy_value())


@typeof_impl.register(pd.DatetimeTZDtype)
def typeof_pd_int_dtype(val, c):
    return PandasDatetimeTZDtype(val.tz)


def get_pytz_type_info(pytz_type):
    """
    Extracts the information used by Bodo when encountering a pytz
    type. This obtains the string name of the zone for most timezones,
    but for FixedOffsets it outputs an integer in nanoseconds.
    """
    if isinstance(pytz_type, pytz._FixedOffset):
        # If we have a fixed offset represent it as an integer
        # offset in ns.
        # Note: pytz_type._offset is a np.timedelta.
        tz_val = pd.Timedelta(pytz_type._offset).value
    else:
        tz_val = pytz_type.zone
        if tz_val not in pytz.all_timezones_set:
            raise BodoError(
                "Unsupported timezone type. Timezones must be a fixedOffset or contain a zone found in pytz.all_timezones"
            )
    return tz_val


def nanoseconds_to_offset(nanoseconds):
    """
    Converts a number of nanoseconds to the appropriate pytz.Offset type.
    """
    num_mins = nanoseconds // (60 * 1000 * 1000 * 1000)
    return pytz.FixedOffset(num_mins)


class DatetimeArrayType(types.IterableType, types.ArrayCompatible):
    """Data type for datetime array with timezones"""

    def __init__(self, tz):
        if isinstance(tz, (pytz._FixedOffset, pytz.tzinfo.BaseTzInfo)):
            tz = get_pytz_type_info(tz)
        if not isinstance(tz, (int, str)):
            raise BodoError(
                "Timezone must be either a valid pytz type with a zone or a fixed offset"
            )
        self.tz = tz
        self._data_array_type = types.Array(types.NPDatetime("ns"), 1, "C")
        self._dtype = PandasDatetimeTZDtype(tz)
        super(DatetimeArrayType, self).__init__(name=f"PandasDatetimeArray[{tz}]")

    @property
    def data_array_type(self):
        return self._data_array_type

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    @property
    def iterator_type(self):
        return BodoArrayIterator(self)

    @property
    def dtype(self):
        return self._dtype

    def copy(self):
        return DatetimeArrayType(self.tz)


@register_model(DatetimeArrayType)
class PandasDatetimeArrayModel(models.StructModel):
    """Datetime array model, storing datetime64 array and timezone"""

    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data_array_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(DatetimeArrayType, "data", "_data")


@typeof_impl.register(pd.arrays.DatetimeArray)
def typeof_pd_datetime_array(val, c):
    if val.tz is None:
        # DatetimeArray can contain no timezone. We don't yet support
        # this yet.
        raise BodoError(
            "Cannot support timezone naive pd.arrays.DatetimeArray. Please convert to a numpy array with .astype('datetime64[ns]')."
        )

    if val.dtype.unit != "ns":
        raise BodoError("Timezone-aware datetime data requires 'ns' units")

    return DatetimeArrayType(val.dtype.tz)


@unbox(DatetimeArrayType)
def unbox_pd_datetime_array(typ, val, c):
    pd_datetime_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    datetime64_str = c.pyapi.string_from_constant_string("datetime64[ns]")
    pd_datetime_arr_obj = c.pyapi.call_method(val, "to_numpy", (datetime64_str,))
    pd_datetime_arr.data = c.unbox(typ.data_array_type, pd_datetime_arr_obj).value

    c.pyapi.decref(pd_datetime_arr_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(pd_datetime_arr._getvalue(), is_error=is_error)


@box(DatetimeArrayType)
def box_pd_datetime_array(typ, val, c):
    """
    We box a the datetime array by extracting the object for the data,
    creating a DatetimeTZDtype from the type string, and finally by
    calling the pandas.arrays.DatetimeArray constructor.
    """
    pd_datetime_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    # Fetch the numpy data array
    # incref since boxing functions steal a reference
    c.context.nrt.incref(c.builder, typ.data_array_type, pd_datetime_arr.data)
    np_arr_obj = c.pyapi.from_native_value(
        typ.data_array_type, pd_datetime_arr.data, c.env_manager
    )

    # Create the timezone type.
    unit_str = c.context.get_constant_generic(c.builder, types.unicode_type, "ns")
    # No need to incref because unit_str is a constant
    unit_str_obj = c.pyapi.from_native_value(
        types.unicode_type, unit_str, c.env_manager
    )
    if isinstance(typ.tz, str):
        tz_str = c.context.get_constant_generic(c.builder, types.unicode_type, typ.tz)
        # No need to incref because tz_str is a constant
        tz_arg_obj = c.pyapi.from_native_value(
            types.unicode_type, tz_str, c.env_manager
        )
    else:
        # We store ns, but the Fixed offset constructor takes minutes.
        offset = nanoseconds_to_offset(typ.tz)
        tz_arg_obj = c.pyapi.unserialize(c.pyapi.serialize_object(offset))

    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    tz_obj = c.pyapi.call_method(
        pd_class_obj, "DatetimeTZDtype", (unit_str_obj, tz_arg_obj)
    )

    # Get the constructor
    pd_array_class_obj = c.pyapi.object_getattr_string(pd_class_obj, "arrays")

    # Call the constructor.
    res = c.pyapi.call_method(pd_array_class_obj, "DatetimeArray", (np_arr_obj, tz_obj))

    c.pyapi.decref(np_arr_obj)
    c.pyapi.decref(unit_str_obj)
    c.pyapi.decref(tz_arg_obj)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(tz_obj)
    c.pyapi.decref(pd_array_class_obj)
    # decref() should be called on native value
    # see https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/boxing.py#L389
    c.context.nrt.decref(c.builder, typ, val)
    return res


@intrinsic
def init_pandas_datetime_array(typingctx, data, tz):
    """
    Initialize a pandas.arrays.DatetimeArray.
    """

    def codegen(context, builder, sig, args):
        data, tz = args

        pd_dt_arr = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        pd_dt_arr.data = data

        context.nrt.incref(builder, sig.args[0], data)

        return pd_dt_arr._getvalue()

    if is_overload_constant_str(tz) or is_overload_constant_int(tz):
        tz_str = get_literal_value(tz)
    else:
        raise BodoError("tz must be a constant string or Fixed Offset")

    return_type = DatetimeArrayType(tz_str)
    sig = return_type(return_type.data_array_type, tz)

    return sig, codegen


# high-level allocation function for tz-aware arrays arrays
@numba.njit(no_cpython_wrapper=True)
def alloc_pd_datetime_array(n, tz):  # pragma: no cover
    data_arr = np.empty(n, dtype="datetime64[ns]")
    return init_pandas_datetime_array(data_arr, tz)


def alloc_pd_datetime_array_equiv(self, scope, equiv_set, loc, args, kws):
    """Array analysis function for alloc_pd_datetime_array() passed to Numba's array analysis
    extension. Assigns output array's size as equivalent to the input size variable.
    """
    assert len(args) == 2 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


ArrayAnalysis._analyze_op_call_bodo_libs_pd_datetime_arr_ext_alloc_pd_datetime_array = (
    alloc_pd_datetime_array_equiv
)


@overload(len, no_unliteral=True)
def overload_pd_datetime_arr_len(A):
    if isinstance(A, DatetimeArrayType):
        return lambda A: len(A._data)  # pragma: no cover


@lower_constant(DatetimeArrayType)
def lower_constant_pd_datetime_arr(context, builder, typ, pyval):
    numpy_data = context.get_constant_generic(
        builder, typ.data_array_type, pyval.to_numpy("datetime64[ns]")
    )
    datetime_arr_val = lir.Constant.literal_struct([numpy_data])
    return datetime_arr_val


@overload_attribute(DatetimeArrayType, "shape")
def overload_pd_datetime_arr_shape(A):
    return lambda A: (len(A._data),)  # pragma: no cover


@overload_attribute(DatetimeArrayType, "nbytes")
def overload_pd_datetime_arr_nbytes(A):
    return lambda A: A._data.nbytes  # pragma: no cover


@overload_method(DatetimeArrayType, "tz_convert", no_unliteral=True)
def overload_pd_datetime_tz_convert(A, tz):
    if tz == types.none:
        # Note this differs from Pandas in the output type.
        # Pandas would still have a DatetimeArrayType with no timezone
        # but we always represent no timezone as datetime64 array.
        def impl(A, tz):  # pragma: no cover
            return A._data.copy()

        return impl

    else:

        def impl(A, tz):  # pragma: no cover
            return init_pandas_datetime_array(A._data.copy(), tz)

    return impl


@overload_method(DatetimeArrayType, "copy", no_unliteral=True)
def overload_pd_datetime_tz_convert(A):
    tz = A.tz

    def impl(A):  # pragma: no cover
        return init_pandas_datetime_array(A._data.copy(), tz)

    return impl


@overload_attribute(DatetimeArrayType, "dtype", no_unliteral=True)
def overload_pd_datetime_dtype(A):
    tz = A.tz
    dtype = pd.DatetimeTZDtype("ns", tz)

    def impl(A):  # pragma: no cover
        return dtype

    return impl


@overload(operator.getitem, no_unliteral=True)
def overload_getitem(A, ind):
    if not isinstance(A, DatetimeArrayType):
        return
    tz = A.tz
    if isinstance(ind, types.Integer):

        def impl(A, ind):  # pragma: no cover
            return bodo.hiframes.pd_timestamp_ext.convert_val_to_timestamp(
                bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A._data[ind]),
                tz,
            )

        return impl

    # bool arr indexing. Note nullable boolean arrays are handled in
    # bool_arr_ind_getitem to ensure NAs are converted to False.
    if (
        ind != bodo.boolean_array
        and is_list_like_index_type(ind)
        and ind.dtype == types.bool_
    ):

        def impl_bool(A, ind):  # pragma: no cover
            ind = bodo.utils.conversion.coerce_to_ndarray(ind)
            new_data = ensure_contig_if_np(A._data[ind])
            return init_pandas_datetime_array(new_data, tz)

        return impl_bool

    # int arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):

        def impl_int_arr(A, ind):  # pragma: no cover
            ind = bodo.utils.conversion.coerce_to_ndarray(ind)
            new_data = ensure_contig_if_np(A._data[ind])
            return init_pandas_datetime_array(new_data, tz)

        return impl_int_arr

    # slice indexing
    if isinstance(ind, types.SliceType):

        def impl_slice(A, ind):  # pragma: no cover
            new_data = ensure_contig_if_np(A._data[ind])
            return init_pandas_datetime_array(new_data, tz)

        return impl_slice

    # This should be the only DatetimeArray implementation
    # except for converting a Nullable boolean index to non-nullable.
    if ind != bodo.boolean_array:  # pragma: no cover
        raise BodoError(
            "operator.getitem with DatetimeArrayType is only supported with an integer index, int arr, boolean array, or slice."
        )


@overload(operator.setitem, no_unliteral=True)
def overload_getitem(A, ind, val):
    if not isinstance(A, DatetimeArrayType):
        return
    tz = A.tz
    if isinstance(ind, types.Integer):
        if not isinstance(val, bodo.PandasTimestampType):  # pragma: no cover
            raise BodoError(
                "operator.setitem with DatetimeArrayType requires a Timestamp value"
            )
        if val.tz != tz:
            raise BodoError(
                "operator.setitem with DatetimeArrayType requires the Timestamp value to share the same timezone"
            )

        def impl(A, ind, val):  # pragma: no cover
            A._data[ind] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(val.value)

        return impl

    raise BodoError(
        "operator.setitem with DatetimeArrayType is only supported with an integer index"
    )


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def unwrap_tz_array(A):
    if isinstance(A, DatetimeArrayType):
        return lambda A: A._data  # pragma: no cover
    return lambda A: A  # pragma: no cover


# array analysis extension
def unwrap_tz_array_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_libs_pd_datetime_arr_ext_unwrap_tz_array = (
    unwrap_tz_array_equiv
)


def create_cmp_op_overload_arr(op):
    """create overload function for comparison operators with pandas timezone aware datetime array"""
    # Import within the function to avoid circular imports
    from bodo.hiframes.pd_timestamp_ext import PandasTimestampType

    def overload_datetime_arr_cmp(lhs, rhs):
        if not (
            isinstance(lhs, DatetimeArrayType) or isinstance(rhs, DatetimeArrayType)
        ):  # pragma: no cover
            # This implementation only handles at least 1 DatetimeArrayType
            return

        # DatetimeArrayType + Scalar tz-aware or date
        if isinstance(lhs, DatetimeArrayType) and (
            isinstance(rhs, PandasTimestampType) or rhs == bodo.datetime_date_type
        ):
            # Note: Checking that tz values match is handled by the scalar comparison.
            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(lhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = op(lhs[i], rhs)
                return out_arr

            return impl

        # Scalar tz-aware or date + DatetimeArrayType.
        elif (
            isinstance(lhs, PandasTimestampType) or lhs == bodo.datetime_date_type
        ) and isinstance(rhs, DatetimeArrayType):
            # Note: Checking that tz values match is handled by the scalar comparison.
            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(rhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(rhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = op(lhs, rhs[i])
                return out_arr

            return impl

        # DatetimeArrayType or date array + DatetimeArrayType or date array
        elif (
            isinstance(lhs, DatetimeArrayType) or lhs == bodo.datetime_date_array_type
        ) and (
            isinstance(rhs, DatetimeArrayType) or rhs == bodo.datetime_date_array_type
        ):
            # Note: Checking that tz values match is handled by the scalar comparison.
            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(
                        lhs, i
                    ) or bodo.libs.array_kernels.isna(rhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = op(lhs[i], rhs[i])
                return out_arr

            return impl

        # Tz-Aware timestamp + Tz-Naive timestamp
        elif isinstance(lhs, DatetimeArrayType) and (
            isinstance(rhs, types.Array) and rhs.dtype == bodo.datetime64ns
        ):
            raise BodoError(
                f"{numba.core.utils.OPERATORS_TO_BUILTINS[op]} with two Timestamps requires both Timestamps share the same timezone. "
                + f"Argument 0 has timezone {lhs.tz} and argument 1 is timezone-naive. "
                + "To compare these values please convert to timezone naive with ts.tz_convert(None)."
            )
        # Tz-Naive timestamp + Tz-Aware timestamp
        elif (
            isinstance(lhs, types.Array) and lhs.dtype == bodo.datetime64ns
        ) and isinstance(rhs, DatetimeArrayType):
            raise BodoError(
                f"{numba.core.utils.OPERATORS_TO_BUILTINS[op]} with two Timestamps requires both Timestamps share the same timezone. "
                + f"Argument 0 is timezone-naive and argument 1 has timezone {rhs.tz}. "
                + "To compare these values please convert to timezone naive with ts.tz_convert(None)."
            )

    return overload_datetime_arr_cmp


def overload_add_operator_datetime_arr(lhs, rhs):
    """
    Implementation for the supported add operations on Timezone-Aware data.
    This function is called from an overload, so it returns an overload.
    This is used for lhs + rhs.

    Either lhs or rhs is assumed to be a DatetimeArrayType based on how this
    function is used.

    Args:
        lhs (types.Type): Bodo type to add. Either (DatetimeArrayType or week_type)
        rhs (types.Type): Bodo type to add. Either (DatetimeArrayType or week_type)

    Raises:
        BodoError: If operator.add is not supported between DatetimeArrayType and the other type.

    Returns:
        func: An implementation function that would be returned from an overload
    """
    if isinstance(lhs, DatetimeArrayType):
        # TODO: Support more types
        if rhs == bodo.week_type:
            tz_literal = lhs.tz

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(lhs)
                out_arr = bodo.libs.pd_datetime_arr_ext.alloc_pd_datetime_array(
                    n, tz_literal
                )
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(lhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = lhs[i] + rhs
                return out_arr

            return impl

        else:
            raise BodoError(
                f"add operator not supported between Timezone-aware timestamp and {rhs}. Please convert to timezone naive with ts.tz_convert(None)"
            )
    else:
        # Note this function is only called if at least one input is a DatetimeArrayType
        # TODO: Support more types
        if lhs == bodo.week_type:

            tz_literal = rhs.tz

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(rhs)
                out_arr = bodo.libs.pd_datetime_arr_ext.alloc_pd_datetime_array(
                    n, tz_literal
                )
                for i in numba.parfors.parfor.internal_prange(n):
                    if bodo.libs.array_kernels.isna(rhs, i):
                        bodo.libs.array_kernels.setna(out_arr, i)
                    else:
                        out_arr[i] = lhs + rhs[i]
                return out_arr

            return impl

        else:
            raise BodoError(
                f"add operator not supported between {lhs} and Timezone-aware timestamp. Please convert to timezone naive with ts.tz_convert(None)"
            )


@register_jitable
def convert_months_offset_to_days(
    curr_year, curr_month, curr_day, num_months
):  # pragma: no cover
    """Converts the number of months to move forward from a current
    year, month, and day into a Timedelta with the appropriate number of days.
    This is used to convert a DateOffset of only months into an equivalent
    pd.Timedelta for us in BodoSQL array kernels

    Args:
        curr_year (types.int64): Current year number
        curr_month (types.int64): Current month number (1-12)
        curr_day (types.int64): Current day number (1-31)
        num_months (types.int64): Number of months to add (either + or -)
    """
    # Account for the 1-indexing in computing the new month
    month_total = (curr_month + num_months) - 1
    new_month = (month_total % 12) + 1
    num_years = month_total // 12
    new_year = curr_year + num_years
    # Make sure the day is still valid in this month, otherwise we truncate
    # to the last day of the month.
    max_day = bodo.hiframes.pd_timestamp_ext.get_days_in_month(new_year, new_month)
    new_day = min(max_day, curr_day)
    curr_ts = pd.Timestamp(year=curr_year, month=curr_month, day=curr_day)
    new_ts = pd.Timestamp(year=new_year, month=new_month, day=new_day)
    return new_ts - curr_ts
