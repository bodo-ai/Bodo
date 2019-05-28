"""
Indexing support for Series objects, including loc/iloc/at/iat types.
"""
import operator
import numpy as np
import pandas as pd
import numba
from numba import types, cgutils
from numba.extending import (models, register_model, lower_cast, infer_getattr,
    type_callable, infer, overload, make_attribute_wrapper, intrinsic,
    overload_attribute)
from numba.extending import (models, register_model, lower_cast, infer_getattr,
    type_callable, infer, overload, make_attribute_wrapper)
from numba.typing.templates import (infer_global, AbstractTemplate, signature,
    AttributeTemplate, bound_function)
import bodo
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.pd_timestamp_ext import (pandas_timestamp_type,
    convert_datetime64_to_timestamp, convert_timestamp_to_datetime64,
    integer_to_dt64)


class SeriesIatType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = "SeriesIatType({})".format(stype)
        super(SeriesIatType, self).__init__(name)



@register_model(SeriesIatType)
class SeriesIatModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('obj', fe_type.stype),
        ]
        super(SeriesIatModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesIatType, 'obj', '_obj')


@intrinsic
def init_series_iat(typingctx, obj=None):

    def codegen(context, builder, signature, args):
        obj_val, = args
        iat_type = signature.return_type

        iat_val = cgutils.create_struct_proxy(iat_type)(context, builder)
        iat_val.obj = obj_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], obj_val)

        return iat_val._getvalue()

    return SeriesIatType(obj)(obj), codegen



@overload_attribute(SeriesType, 'iat')
def overload_series_iat(s):
    return lambda s: bodo.hiframes.series_indexing.init_series_iat(s)


@overload(operator.getitem)
def overload_series_iat_getitem(I, idx):
    if isinstance(I, SeriesIatType):
        if not isinstance(idx, types.Integer):
            raise ValueError(
                'iAt based indexing can only have integer indexers')

        # box dt64 to timestamp
        if I.stype.dtype == types.NPDatetime('ns'):
            return (lambda I, idx: convert_datetime64_to_timestamp(
                    np.int64(bodo.hiframes.api.get_series_data(I._obj)[idx])))

        # TODO: box timedelta64, datetime.datetime/timedelta
        return lambda I, idx: bodo.hiframes.api.get_series_data(I._obj)[idx]


@overload(operator.setitem)
def overload_series_iat_setitem(I, idx, val):
    if isinstance(I, SeriesIatType):
        if not isinstance(idx, types.Integer):
            raise ValueError(
                'iAt based indexing can only have integer indexers')
        # check string setitem
        if I.stype.dtype == bodo.string_type:
            raise ValueError("Series string setitem not supported yet")
        # unbox dt64 from Timestamp (TODO: timedelta and other datetimelike)
        # see unboxing pandas/core/arrays/datetimes.py:
        # DatetimeArray._unbox_scalar
        if (I.stype.dtype == types.NPDatetime('ns')
                and val == pandas_timestamp_type):
            def impl_dt(I, idx, val):
                s = integer_to_dt64(convert_timestamp_to_datetime64(val))
                bodo.hiframes.api.get_series_data(I._obj)[idx] = s
            return impl_dt

        def impl(I, idx, val):
            bodo.hiframes.api.get_series_data(I._obj)[idx] = val

        return impl
