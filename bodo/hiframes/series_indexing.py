# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Indexing support for Series objects, including loc/iloc/at/iat types.
"""
import operator
import numpy as np
import pandas as pd
import numba
from numba import types, cgutils
from numba.extending import (
    models,
    register_model,
    lower_cast,
    infer_getattr,
    type_callable,
    infer,
    overload,
    make_attribute_wrapper,
    intrinsic,
    overload_attribute,
)
from numba.extending import (
    models,
    register_model,
    lower_cast,
    infer_getattr,
    type_callable,
    infer,
    overload,
    make_attribute_wrapper,
)
from numba.typing.templates import (
    infer_global,
    AbstractTemplate,
    signature,
    AttributeTemplate,
    bound_function,
)
import bodo
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.pd_timestamp_ext import (
    pandas_timestamp_type,
    convert_datetime64_to_timestamp,
    convert_timestamp_to_datetime64,
    integer_to_dt64,
)
from bodo.hiframes.pd_index_ext import NumericIndexType, RangeIndexType
from bodo.utils.typing import is_list_like_index_type


##############################  iat  ######################################


class SeriesIatType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = "SeriesIatType({})".format(stype)
        super(SeriesIatType, self).__init__(name)


@register_model(SeriesIatType)
class SeriesIatModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.stype)]
        super(SeriesIatModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesIatType, "obj", "_obj")


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


@overload_attribute(SeriesType, "iat")
def overload_series_iat(s):
    return lambda s: bodo.hiframes.series_indexing.init_series_iat(s)


@overload(operator.getitem)
def overload_series_iat_getitem(I, idx):
    if isinstance(I, SeriesIatType):
        if not isinstance(idx, types.Integer):
            raise ValueError("iAt based indexing can only have integer indexers")

        # box dt64 to timestamp
        if I.stype.dtype == types.NPDatetime("ns"):
            return lambda I, idx: convert_datetime64_to_timestamp(
                np.int64(bodo.hiframes.api.get_series_data(I._obj)[idx])
            )

        # TODO: box timedelta64, datetime.datetime/timedelta
        return lambda I, idx: bodo.hiframes.api.get_series_data(I._obj)[idx]


@overload(operator.setitem)
def overload_series_iat_setitem(I, idx, val):
    if isinstance(I, SeriesIatType):
        if not isinstance(idx, types.Integer):
            raise ValueError("iAt based indexing can only have integer indexers")
        # check string setitem
        if I.stype.dtype == bodo.string_type:
            raise ValueError("Series string setitem not supported yet")
        # unbox dt64 from Timestamp (TODO: timedelta and other datetimelike)
        # see unboxing pandas/core/arrays/datetimes.py:
        # DatetimeArray._unbox_scalar
        if I.stype.dtype == types.NPDatetime("ns") and val == pandas_timestamp_type:

            def impl_dt(I, idx, val):
                s = integer_to_dt64(convert_timestamp_to_datetime64(val))
                bodo.hiframes.api.get_series_data(I._obj)[idx] = s

            return impl_dt

        def impl(I, idx, val):
            bodo.hiframes.api.get_series_data(I._obj)[idx] = val

        return impl


##############################  iloc  ######################################


class SeriesIlocType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = "SeriesIlocType({})".format(stype)
        super(SeriesIlocType, self).__init__(name)


@register_model(SeriesIlocType)
class SeriesIlocModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.stype)]
        super(SeriesIlocModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesIlocType, "obj", "_obj")


@intrinsic
def init_series_iloc(typingctx, obj=None):
    def codegen(context, builder, signature, args):
        obj_val, = args
        iloc_type = signature.return_type

        iloc_val = cgutils.create_struct_proxy(iloc_type)(context, builder)
        iloc_val.obj = obj_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], obj_val)

        return iloc_val._getvalue()

    return SeriesIlocType(obj)(obj), codegen


@overload_attribute(SeriesType, "iloc")
def overload_series_iloc(s):
    return lambda s: bodo.hiframes.series_indexing.init_series_iloc(s)


@overload(operator.getitem)
def overload_series_iloc_getitem(I, idx):
    if isinstance(I, SeriesIlocType):
        # Integer case returns scalar
        if isinstance(idx, types.Integer):
            # box dt64 to timestamp
            # TODO: box timedelta64, datetime.datetime/timedelta
            return lambda I, idx: bodo.utils.conversion.box_if_dt64(
                bodo.hiframes.api.get_series_data(I._obj)[idx]
            )

        # all other cases return a Series
        # list of ints or array of ints
        # list of bools or array of bools
        # TODO: fix list of int getitem on Arrays in Numba
        # TODO: fix none Index
        # TODO: other list-like such as Series/Index
        if is_list_like_index_type(idx) and isinstance(
            idx.dtype, (types.Integer, types.Boolean)
        ):

            def impl(I, idx):
                idx_t = bodo.utils.conversion.coerce_to_ndarray(idx)
                arr = bodo.hiframes.api.get_series_data(I._obj)[idx_t]
                index = bodo.utils.conversion.fix_none_index(
                    bodo.hiframes.api.get_series_index(I._obj), len(arr)
                )[idx_t]
                name = bodo.hiframes.api.get_series_name(I._obj)
                return bodo.hiframes.pd_series_ext.init_series(arr, index, name)

            return impl

        # slice
        if isinstance(idx, types.SliceType):

            def impl(I, idx):
                arr = bodo.hiframes.api.get_series_data(I._obj)[idx]
                index = bodo.hiframes.api.get_series_index(I._obj)[idx]
                name = bodo.hiframes.api.get_series_name(I._obj)
                return bodo.hiframes.pd_series_ext.init_series(arr, index, name)

            return impl

        raise ValueError("iloc[] getitem using {} not supported".format(idx))


@overload(operator.setitem)
def overload_series_iloc_setitem(I, idx, val):
    if isinstance(I, SeriesIlocType):
        # check string setitem
        if I.stype.dtype == bodo.string_type:
            raise ValueError("Series string setitem not supported yet")

        # integer case same as iat
        if isinstance(idx, types.Integer):
            # unbox dt64 from Timestamp (TODO: timedelta and other datetimelike)
            if I.stype.dtype == types.NPDatetime("ns") and val == pandas_timestamp_type:

                def impl_dt(I, idx, val):
                    s = integer_to_dt64(convert_timestamp_to_datetime64(val))
                    bodo.hiframes.api.get_series_data(I._obj)[idx] = s

                return impl_dt

            def impl(I, idx, val):
                bodo.hiframes.api.get_series_data(I._obj)[idx] = val

            return impl

        # all other cases just set data
        # slice
        if isinstance(idx, types.SliceType):

            def impl_slice(I, idx, val):
                bodo.hiframes.api.get_series_data(I._obj)[
                    idx
                ] = bodo.utils.conversion.coerce_to_array(val, False)

            return impl_slice

        # list of ints or array of ints
        # list of bools or array of bools
        # TODO: fix list of int getitem on Arrays in Numba
        if is_list_like_index_type(idx) and isinstance(
            idx.dtype, (types.Integer, types.Boolean)
        ):

            def impl_arr(I, idx, val):
                idx_t = bodo.utils.conversion.coerce_to_ndarray(idx)
                bodo.hiframes.api.get_series_data(I._obj)[
                    idx_t
                ] = bodo.utils.conversion.coerce_to_array(val, False)

            return impl_arr

        raise ValueError("iloc[] setitem using {} not supported".format(idx))


######################## __getitem__/__setitem__ ########################


@overload(operator.getitem)
def overload_series_getitem(S, idx):
    # XXX: Series getitem performs both label-based and location-based indexing
    if isinstance(S, SeriesType):
        # Integer index is location unless if Index is integer
        if isinstance(idx, types.Integer):
            # integer Index not supported yet
            if isinstance(S.index, NumericIndexType) and isinstance(
                S.index.dtype, types.Integer
            ):
                raise ValueError(
                    "Indexing Series with Integer index using []"
                    " (which is label-based) not supported yet"
                )
            if isinstance(S.index, RangeIndexType):
                # TODO: check for invalid idx
                # TODO: test different RangeIndex cases
                def impl(S, idx):
                    arr = bodo.hiframes.api.get_series_data(S)
                    I = bodo.hiframes.api.get_series_index(S)
                    idx_t = idx * I._step + I._start
                    return bodo.utils.conversion.box_if_dt64(arr[idx_t])

                return impl

            # other indices are just ignored and location returned
            return lambda S, idx: bodo.utils.conversion.box_if_dt64(
                bodo.hiframes.api.get_series_data(S)[idx]
            )

        # TODO: other list-like such as Series, Index
        if is_list_like_index_type(idx) and isinstance(
            idx.dtype, (types.Integer, types.Boolean)
        ):
            if (
                isinstance(S.index, NumericIndexType)
                and isinstance(S.index.dtype, types.Integer)
                and isinstance(idx.dtype, types.Integer)
            ):
                raise ValueError(
                    "Indexing Series with Integer index using []"
                    " (which is label-based) not supported yet"
                )

            def impl_arr(S, idx):
                idx_t = bodo.utils.conversion.coerce_to_ndarray(idx)
                arr = bodo.hiframes.api.get_series_data(S)[idx_t]
                index = bodo.utils.conversion.fix_none_index(
                    bodo.hiframes.api.get_series_index(S), len(arr)
                )[idx_t]
                name = bodo.hiframes.api.get_series_name(S)
                return bodo.hiframes.pd_series_ext.init_series(arr, index, name)

            return impl_arr

        # slice
        if isinstance(idx, types.SliceType):
            # TODO: fix none Index
            # XXX: slices are only integer in Numba?
            # TODO: support label slices like '2015-03-21':'2015-03-24'
            def impl_slice(S, idx):
                arr = bodo.hiframes.api.get_series_data(S)[idx]
                index = bodo.hiframes.api.get_series_index(S)[idx]
                name = bodo.hiframes.api.get_series_name(S)
                return bodo.hiframes.pd_series_ext.init_series(arr, index, name)

            return impl_slice

        # TODO: handle idx as SeriesType on array
        raise ValueError("setting Series value using {} not supported yet".format(idx))


@overload(operator.setitem)
def overload_series_setitem(S, idx, val):
    if isinstance(S, SeriesType):
        # check string setitem
        if S.dtype == bodo.string_type:
            raise ValueError("Series string setitem not supported yet")

        # integer case same as iat
        if isinstance(idx, types.Integer):
            if isinstance(S.index, NumericIndexType) and isinstance(
                S.index.dtype, types.Integer
            ):
                raise ValueError(
                    "Indexing Series with Integer index using []"
                    " (which is label-based) not supported yet"
                )
            # unbox dt64 from Timestamp (TODO: timedelta and other datetimelike)
            if S.dtype == types.NPDatetime("ns") and val == pandas_timestamp_type:

                def impl_dt(S, idx, val):
                    s = integer_to_dt64(convert_timestamp_to_datetime64(val))
                    bodo.hiframes.api.get_series_data(S)[idx] = s

                return impl_dt

            def impl(S, idx, val):
                bodo.hiframes.api.get_series_data(S)[idx] = val

            return impl

        # all other cases just set data
        # slice
        if isinstance(idx, types.SliceType):

            def impl_slice(S, idx, val):
                bodo.hiframes.api.get_series_data(S)[
                    idx
                ] = bodo.utils.conversion.coerce_to_array(val, False)

            return impl_slice

        # list of ints or array of ints
        # list of bools or array of bools
        # TODO: fix list of int getitem on Arrays in Numba
        if is_list_like_index_type(idx) and isinstance(
            idx.dtype, (types.Integer, types.Boolean)
        ):
            if (
                isinstance(S.index, NumericIndexType)
                and isinstance(S.index.dtype, types.Integer)
                and isinstance(idx.dtype, types.Integer)
            ):
                raise ValueError(
                    "Indexing Series with Integer index using []"
                    " (which is label-based) not supported yet"
                )

            def impl_arr(S, idx, val):
                idx_t = bodo.utils.conversion.coerce_to_ndarray(idx)
                bodo.hiframes.api.get_series_data(S)[
                    idx_t
                ] = bodo.utils.conversion.coerce_to_array(val, False)

            return impl_arr

        raise ValueError("Series [] setitem using {} not supported".format(idx))
