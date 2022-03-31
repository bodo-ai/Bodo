# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""DatetimeArray extension for Pandas DatetimeArray with timezone support."""

import operator
from re import T

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
    register_model,
    typeof_impl,
    unbox,
)

import bodo
from bodo.utils.conversion import ensure_contig_if_np
from bodo.utils.typing import (
    BodoArrayIterator,
    BodoError,
    get_literal_value,
    get_overload_const_bool,
    is_list_like_index_type,
    is_overload_constant_bool,
    is_overload_constant_int,
    is_overload_constant_str,
    raise_bodo_error,
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


@overload(len, no_unliteral=True)
def overload_pd_datetime_arr_len(A):
    if isinstance(A, DatetimeArrayType):
        return lambda A: len(A._data)  # pragma: no cover


@lower_constant(DatetimeArrayType)
def lower_constant_pd_datetime_arr(context, builder, typ, pyval):
    raise BodoError("pd.DatetimeArray(): Lowering not supported")


@overload_attribute(DatetimeArrayType, "shape")
def overload_pd_datetime_arr_shape(A):
    return lambda A: (len(A._data),)  # pragma: no cover


@overload_method(DatetimeArrayType, "tz_convert", no_unliteral=True)
def overload_pd_datetime_tz_convert(A, tz):
    if tz == types.none:
        raise_bodo_error("tz_convert(): tz must be a string or Fixed Offset")

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

    # bool arr indexing
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:

        def impl_bool(A, ind):  # pragma: no cover
            ind = bodo.utils.conversion.coerce_to_ndarray(ind)
            new_data = ensure_contig_if_np(A._data[ind])
            return init_pandas_datetime_array(new_data, tz)

        return impl_bool

    # slice indexing
    if isinstance(ind, types.SliceType):

        def impl_slice(A, ind):  # pragma: no cover
            new_data = ensure_contig_if_np(A._data[ind])
            return init_pandas_datetime_array(new_data, tz)

        return impl_slice

    raise BodoError(
        "operator.getitem with DatetimeArrayType is only supported with an integer index, boolean array, or slice."
    )


# TODO: Add an alias?
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def unwrap_tz_array(A):
    if isinstance(A, DatetimeArrayType):
        return lambda A: A._data  # pragma: no cover
    return lambda A: A  # pragma: no cover
