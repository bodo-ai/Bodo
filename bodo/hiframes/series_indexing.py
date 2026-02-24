"""
Indexing support for Series objects, including loc/iloc/at/iat types.
"""

import operator

import numpy as np
from numba.core import cgutils, types
from numba.extending import (
    intrinsic,
    lower_cast,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    register_model,
)

import bodo
from bodo.hiframes.datetime_timedelta_ext import (
    datetime_timedelta_type,
    pd_timedelta_type,
)
from bodo.hiframes.pd_index_ext import (
    HeterogeneousIndexType,
    NumericIndexType,
    RangeIndexType,
)
from bodo.hiframes.pd_series_ext import HeterogeneousSeriesType, SeriesType
from bodo.hiframes.pd_timestamp_ext import (
    convert_datetime64_to_timestamp,
    convert_numpy_timedelta64_to_pd_timedelta,
    integer_to_dt64,
    pd_timestamp_tz_naive_type,
)
from bodo.utils.typing import (
    BodoError,
    get_literal_value,
    get_overload_const_int,
    get_overload_const_tuple,
    is_immutable_array,
    is_list_like_index_type,
    is_literal_type,
    is_overload_constant_int,
    is_overload_constant_str,
    is_overload_constant_tuple,
    is_scalar_type,
    raise_bodo_error,
)

##############################  iat  ######################################


class SeriesIatType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = f"SeriesIatType({stype})"
        super().__init__(name)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    ndim = 1


@register_model(SeriesIatType)
class SeriesIatModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.stype)]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesIatType, "obj", "_obj")


@intrinsic
def init_series_iat(typingctx, obj):
    def codegen(context, builder, signature, args):
        (obj_val,) = args
        iat_type = signature.return_type

        iat_val = cgutils.create_struct_proxy(iat_type)(context, builder)
        iat_val.obj = obj_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], obj_val)

        return iat_val._getvalue()

    return SeriesIatType(obj)(obj), codegen


@overload_attribute(SeriesType, "iat")
def overload_series_iat(s):
    return lambda s: bodo.hiframes.series_indexing.init_series_iat(s)


@overload(operator.getitem, no_unliteral=True)
def overload_series_iat_getitem(I, idx):
    if isinstance(I, SeriesIatType):
        if not isinstance(types.unliteral(idx), types.Integer):
            raise BodoError("iAt based indexing can only have integer indexers")

        # box dt64 to timestamp
        if I.stype.dtype == types.NPDatetime("ns"):
            return lambda I, idx: convert_datetime64_to_timestamp(
                np.int64(bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx])
            )

        # box dt64 to pd.Timedelta
        if I.stype.dtype == types.NPTimedelta("ns"):
            return lambda I, idx: convert_numpy_timedelta64_to_pd_timedelta(
                bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx]
            )

        return lambda I, idx: bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx]


@overload(operator.setitem, no_unliteral=True)
def overload_series_iat_setitem(I, idx, val):
    if isinstance(I, SeriesIatType):
        if not isinstance(idx, types.Integer):
            raise BodoError("iAt based indexing can only have integer indexers")
        # check string/binary setitem
        if I.stype.dtype == bodo.types.string_type and val is not types.none:
            raise BodoError("Series string setitem not supported yet")
        if I.stype.dtype == bodo.types.bytes_type and val is not types.none:
            raise BodoError("Series binary setitem not supported yet")

        # Bodo Restriction, cannot set item with immutable array.
        if is_immutable_array(I.stype.data):
            raise BodoError(
                f"Series setitem not supported for Series with immutable array type {I.stype.data}"
            )
        # unbox dt64 from Timestamp
        # see unboxing pandas/core/arrays/datetimes.py:
        # DatetimeArray._unbox_scalar
        if (
            I.stype.dtype == types.NPDatetime("ns")
            and val == pd_timestamp_tz_naive_type
        ):

            def impl_dt(I, idx, val):  # pragma: no cover
                s = integer_to_dt64(val.value)
                bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx] = s

            return impl_dt

        # unbox dt64 from datetime.datetime
        if (
            I.stype.dtype == types.NPDatetime("ns")
            and val == bodo.types.datetime_datetime_type
        ):

            def impl_dt(I, idx, val):  # pragma: no cover
                s = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(val)
                bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx] = s

            return impl_dt

        # unbox td64 from datetime.timedelta
        if I.stype.dtype == types.NPTimedelta("ns") and val == datetime_timedelta_type:

            def impl_dt(I, idx, val):  # pragma: no cover
                val_b = bodo.hiframes.datetime_timedelta_ext._to_nanoseconds(val)
                s = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val_b)
                bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx] = s

            return impl_dt

        # unbox td64 from datetime.Timedelta
        if I.stype.dtype == types.NPTimedelta("ns") and val == pd_timedelta_type:

            def impl_dt(I, idx, val):  # pragma: no cover
                val_b = val.value
                s = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val_b)
                bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx] = s

            return impl_dt

        def impl(I, idx, val):  # pragma: no cover
            bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx] = val

        return impl


##############################  iloc  ######################################


class SeriesIlocType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = f"SeriesIlocType({stype})"
        super().__init__(name)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    ndim = 1


@register_model(SeriesIlocType)
class SeriesIlocModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.stype)]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesIlocType, "obj", "_obj")


@intrinsic
def init_series_iloc(typingctx, obj):
    def codegen(context, builder, signature, args):
        (obj_val,) = args
        iloc_type = signature.return_type

        iloc_val = cgutils.create_struct_proxy(iloc_type)(context, builder)
        iloc_val.obj = obj_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], obj_val)

        return iloc_val._getvalue()

    return SeriesIlocType(obj)(obj), codegen


@overload_attribute(SeriesType, "iloc")
def overload_series_iloc(s):
    return lambda s: bodo.hiframes.series_indexing.init_series_iloc(s)


@overload(operator.getitem, no_unliteral=True)
def overload_series_iloc_getitem(I, idx):
    if isinstance(I, SeriesIlocType):
        # convert dt64 to pd.timedelta
        if I.stype.dtype == types.NPTimedelta("ns") and isinstance(idx, types.Integer):
            return lambda I, idx: convert_numpy_timedelta64_to_pd_timedelta(
                bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx]
            )

        # Integer case returns scalar
        if isinstance(idx, types.Integer):
            # box dt64 to timestamp

            # TODO: box timedelta64, datetime.datetime/timedelta
            return lambda I, idx: bodo.utils.conversion.box_if_dt64(
                bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx]
            )

        # all other cases return a Series
        # list of ints or array of ints
        # list of bools or array of bools
        # TODO: fix list of int getitem on Arrays in Numba
        # so we can unify the slice and list implementations
        # TODO: other list-like such as Series/Index
        if is_list_like_index_type(idx) and isinstance(
            idx.dtype, (types.Integer, types.Boolean)
        ):

            def impl(I, idx):  # pragma: no cover
                S = I._obj
                # This has a separate implementation because numpy arrays
                # cannot support list of int getitem and we must first
                # convert to an array
                idx_t = bodo.utils.conversion.coerce_to_array(idx)
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)[idx_t]
                index = bodo.hiframes.pd_series_ext.get_series_index(S)[idx_t]
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                return bodo.hiframes.pd_series_ext.init_series(arr, index, name)

            return impl

        # slice
        if isinstance(idx, types.SliceType):

            def impl_slice(I, idx):  # pragma: no cover
                S = I._obj
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)[idx]
                index = bodo.hiframes.pd_series_ext.get_series_index(S)[idx]
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                return bodo.hiframes.pd_series_ext.init_series(arr, index, name)

            return impl_slice

        # TODO: error-checking test
        raise BodoError(
            f"Series.iloc[] getitem using {idx} not supported"
        )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def overload_series_iloc_setitem(I, idx, val):
    if isinstance(I, SeriesIlocType):
        # check string/binary setitem
        if I.stype.dtype == bodo.types.string_type and val is not types.none:
            raise BodoError(
                "Series string setitem not supported yet"
            )  # pragma: no cover
        if I.stype.dtype == bodo.types.bytes_type and val is not types.none:
            raise BodoError(
                "Series binary setitem not supported yet"
            )  # pragma: no cover

        # Bodo Restriction, cannot set item with immutable array.
        if is_immutable_array(I.stype.data):
            raise BodoError(
                f"Series setitem not supported for Series with immutable array type {I.stype.data}"
            )

        # integer case same as iat
        # Scalar val is the same as integer case.
        if isinstance(idx, types.Integer) or (
            isinstance(idx, types.SliceType) and is_scalar_type(val)
        ):
            # unbox dt64 from Timestamp (TODO: timedelta and other datetimelike)
            if (
                I.stype.dtype == types.NPDatetime("ns")
                and val == pd_timestamp_tz_naive_type
            ):

                def impl_dt(I, idx, val):  # pragma: no cover
                    s = integer_to_dt64(val.value)
                    bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx] = s

                return impl_dt

            if (
                I.stype.dtype == types.NPDatetime("ns")
                and val == bodo.types.datetime_datetime_type
            ):

                def impl_dt(I, idx, val):  # pragma: no cover
                    s = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(val)
                    bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx] = s

                return impl_dt

            if (
                I.stype.dtype == types.NPTimedelta("ns")
                and val == datetime_timedelta_type
            ):

                def impl_dt(I, idx, val):  # pragma: no cover
                    val_b = bodo.hiframes.datetime_timedelta_ext._to_nanoseconds(val)
                    s = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val_b)
                    bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx] = s

                return impl_dt

            if I.stype.dtype == types.NPTimedelta("ns") and val == pd_timedelta_type:

                def impl_dt(I, idx, val):  # pragma: no cover
                    val_b = val.value
                    s = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val_b)
                    bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx] = s

                return impl_dt

            def impl(I, idx, val):  # pragma: no cover
                bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx] = val

            return impl

        # all other cases just set data
        # slice
        if isinstance(idx, types.SliceType):

            def impl_slice(I, idx, val):  # pragma: no cover
                bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx] = (
                    bodo.utils.conversion.coerce_to_array(val, False)
                )

            return impl_slice

        # list of ints or array of ints
        # list of bools or array of bools
        # TODO: fix list of int getitem on Arrays in Numba
        if is_list_like_index_type(idx) and isinstance(
            idx.dtype, (types.Integer, types.Boolean)
        ):
            # Scalar case the same as int/slice. Needs a separate
            # implementation for bodo.utils.conversion.coerce_to_array
            if is_scalar_type(val):
                if (
                    I.stype.dtype == types.NPDatetime("ns")
                    and val == pd_timestamp_tz_naive_type
                ):

                    def impl_dt(I, idx, val):  # pragma: no cover
                        s = integer_to_dt64(val.value)
                        idx_t = bodo.utils.conversion.coerce_to_array(idx)
                        bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx_t] = s

                    return impl_dt

                if (
                    I.stype.dtype == types.NPDatetime("ns")
                    and val == bodo.types.datetime_datetime_type
                ):

                    def impl_dt(I, idx, val):  # pragma: no cover
                        s = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(
                            val
                        )
                        idx_t = bodo.utils.conversion.coerce_to_array(idx)
                        bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx_t] = s

                    return impl_dt

                if (
                    I.stype.dtype == types.NPTimedelta("ns")
                    and val == datetime_timedelta_type
                ):

                    def impl_dt(I, idx, val):  # pragma: no cover
                        val_b = bodo.hiframes.datetime_timedelta_ext._to_nanoseconds(
                            val
                        )
                        s = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val_b)
                        idx_t = bodo.utils.conversion.coerce_to_array(idx)
                        bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx_t] = s

                    return impl_dt

                if (
                    I.stype.dtype == types.NPTimedelta("ns")
                    and val == pd_timedelta_type
                ):

                    def impl_dt(I, idx, val):  # pragma: no cover
                        val_b = val.value
                        s = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val_b)
                        idx_t = bodo.utils.conversion.coerce_to_array(idx)
                        bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx_t] = s

                    return impl_dt

                def impl(I, idx, val):  # pragma: no cover
                    idx_t = bodo.utils.conversion.coerce_to_array(idx)
                    bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx_t] = val

                return impl

            def impl_arr(I, idx, val):  # pragma: no cover
                idx_t = bodo.utils.conversion.coerce_to_array(idx)
                bodo.hiframes.pd_series_ext.get_series_data(I._obj)[idx_t] = (
                    bodo.utils.conversion.coerce_to_array(val, False)
                )

            return impl_arr

        # TODO: error-checking test
        raise BodoError(
            f"Series.iloc[] setitem using {idx} not supported"
        )  # pragma: no cover


###############################  loc  ######################################


class SeriesLocType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = f"SeriesLocType({stype})"
        super().__init__(name)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    ndim = 1


@register_model(SeriesLocType)
class SeriesLocModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.stype)]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesLocType, "obj", "_obj")


@intrinsic
def init_series_loc(typingctx, obj):
    def codegen(context, builder, signature, args):
        (obj_val,) = args
        loc_type = signature.return_type

        loc_val = cgutils.create_struct_proxy(loc_type)(context, builder)
        loc_val.obj = obj_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], obj_val)

        return loc_val._getvalue()

    return SeriesLocType(obj)(obj), codegen


@overload_attribute(SeriesType, "loc")
def overload_series_loc(s):
    return lambda s: bodo.hiframes.series_indexing.init_series_loc(s)


@overload(operator.getitem)
def overload_series_loc_getitem(I, idx):
    if not isinstance(I, SeriesLocType):
        return

    # S.loc[cond]
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:

        def impl(I, idx):  # pragma: no cover
            S = I._obj
            idx_t = bodo.utils.conversion.coerce_to_array(idx)
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)[idx_t]
            index = bodo.hiframes.pd_series_ext.get_series_index(S)[idx_t]
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            return bodo.hiframes.pd_series_ext.init_series(arr, index, name)

        return impl

    # TODO: [BE-122] Throw an Error if idx isn't within the range.
    # Pandas throws a KeyError

    # int label from RangeIndex, e.g. S.loc[3]
    if isinstance(idx, types.Integer) and isinstance(I.stype.index, RangeIndexType):

        def impl_range_index_int(I, idx):  # pragma: no cover
            S = I._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            # calculate array index from label value
            arr_idx = (idx - index._start) // index._step
            return arr[arr_idx]

        return impl_range_index_int

    # TODO: error-checking test
    raise BodoError(
        f"Series.loc[] getitem (location-based indexing) using {idx} not supported yet"
    )  # pragma: no cover


@overload(operator.setitem)
def overload_series_loc_setitem(I, idx, val):
    if not isinstance(I, SeriesLocType):
        return

    if is_immutable_array(I.stype.data):
        raise BodoError(
            f"Series setitem not supported for Series with immutable array type {I.stype.data}"
        )

    # S.loc[cond]
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:
        # Series.loc[] setitem with boolean array is same as Series[] setitem
        def impl(I, idx, val):  # pragma: no cover
            S = I._obj
            S[idx] = val

        return impl

    raise BodoError(
        f"Series.loc[] setitem (location-based indexing) using {idx} not supported yet"
    )  # pragma: no cover


######################## __getitem__/__setitem__ ########################


@overload(operator.getitem)
def overload_series_getitem(S, idx):
    # XXX: Series getitem performs both label-based and location-based indexing
    # If we have a HeterogeneousIndexType with a constant tuple, then we want
    # to use a different overload (see overload_const_index_series_getitem)
    if isinstance(S, SeriesType) and not (
        isinstance(S.index, HeterogeneousIndexType)
        and is_overload_constant_tuple(S.index.data)
    ):
        # Integer index is location unless if Index is integer
        if isinstance(idx, types.Integer):
            # integer Index not supported yet
            if isinstance(S.index, NumericIndexType) and isinstance(
                S.index.dtype, types.Integer
            ):
                raise BodoError(
                    "Indexing Series with Integer index using []"
                    " (which is label-based) not supported yet"
                )
            if isinstance(S.index, RangeIndexType):
                # TODO: check for invalid idx
                # TODO: test different RangeIndex cases
                def impl(S, idx):  # pragma: no cover
                    arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                    I = bodo.hiframes.pd_series_ext.get_series_index(S)
                    idx_t = idx * I._step + I._start
                    return bodo.utils.conversion.box_if_dt64(arr[idx_t])

                return impl

            # other indices are just ignored and location returned
            return lambda S, idx: bodo.utils.conversion.box_if_dt64(
                bodo.hiframes.pd_series_ext.get_series_data(S)[idx]
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
                raise BodoError(
                    "Indexing Series with Integer index using []"
                    " (which is label-based) not supported yet"
                )

            def impl_arr(S, idx):  # pragma: no cover
                idx_t = bodo.utils.conversion.coerce_to_array(idx)
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)[idx_t]
                index = bodo.hiframes.pd_series_ext.get_series_index(S)[idx_t]
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                return bodo.hiframes.pd_series_ext.init_series(arr, index, name)

            return impl_arr

        # slice
        if isinstance(idx, types.SliceType):
            # TODO: fix none Index
            # XXX: slices are only integer in Numba?
            # TODO: support label slices like '2015-03-21':'2015-03-24'
            def impl_slice(S, idx):  # pragma: no cover
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)[idx]
                index = bodo.hiframes.pd_series_ext.get_series_index(S)[idx]
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                return bodo.hiframes.pd_series_ext.init_series(arr, index, name)

            return impl_slice

        # pragma is needed, because we don't check the immediately
        # following error in a non-slow test.
        if idx == bodo.types.string_type or is_overload_constant_str(
            idx
        ):  # pragma: no cover
            # TODO: throw an error if not distributed, BE-1535/BE-1536
            if isinstance(S.index, bodo.hiframes.pd_index_ext.StringIndexType):

                def impl_str_getitem(S, idx):  # pragma: no cover
                    series_idx = bodo.hiframes.pd_series_ext.get_series_index(S)
                    # get_loc currently returns a optional index, or throws an error if there's duplicates
                    # this will need to be changed if get_loc supports duplicate labels BE-1562
                    real_idx = series_idx.get_loc(idx)

                    if real_idx is None:
                        raise IndexError
                    else:
                        val = bodo.hiframes.pd_series_ext.get_series_data(S)[
                            bodo.utils.indexing.unoptional(real_idx)
                        ]
                    return val

                return impl_str_getitem
            else:
                raise BodoError(
                    "Cannot get Series value using a string, unless the index type is also string"
                )

        # TODO: handle idx as SeriesType on array
        raise BodoError(f"getting Series value using {idx} not supported yet")

    # convert Series index on Array getitem to array
    elif bodo.utils.utils.is_array_typ(S) and isinstance(idx, SeriesType):
        return lambda S, idx: S[bodo.hiframes.pd_series_ext.get_series_data(idx)]


@overload(operator.setitem, no_unliteral=True)
def overload_series_setitem(S, idx, val):
    if isinstance(S, SeriesType):
        # check string setitem
        if (
            S.dtype == bodo.types.string_type
            and val is not types.none
            and not (is_list_like_index_type(idx) and idx.dtype == types.bool_)
        ):
            raise BodoError("Series string setitem not supported yet")
        elif S.dtype == bodo.types.bytes_type:
            # NOTE: we can loosen the above restriction to be the same as the string
            # array restriction, if we implement boolean list index setitem
            # on the underlying binary array type.
            raise BodoError("Series binary setitem not supported yet")

        # integer case same as iat
        if isinstance(idx, types.Integer):
            if isinstance(S.index, NumericIndexType) and isinstance(
                S.index.dtype, types.Integer
            ):
                raise BodoError(
                    "Indexing Series with Integer index using []"
                    " (which is label-based) not supported yet"
                )
            # unbox dt64 from Timestamp (TODO: timedelta and other datetimelike)
            if S.dtype == types.NPDatetime("ns") and val == pd_timestamp_tz_naive_type:

                def impl_dt(S, idx, val):  # pragma: no cover
                    s = integer_to_dt64(val.value)
                    bodo.hiframes.pd_series_ext.get_series_data(S)[idx] = s

                return impl_dt

            if S.dtype == types.NPTimedelta("ns") and val == datetime_timedelta_type:

                def impl_dt(S, idx, val):  # pragma: no cover
                    val_b = bodo.hiframes.datetime_timedelta_ext._to_nanoseconds(val)
                    s = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val_b)
                    bodo.hiframes.pd_series_ext.get_series_data(S)[idx] = s

                return impl_dt

            if S.dtype == types.NPTimedelta("ns") and val == pd_timedelta_type:

                def impl_dt(S, idx, val):  # pragma: no cover
                    val_b = val.value
                    s = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(val_b)
                    bodo.hiframes.pd_series_ext.get_series_data(S)[idx] = s

                return impl_dt

            def impl(S, idx, val):  # pragma: no cover
                bodo.hiframes.pd_series_ext.get_series_data(S)[idx] = val

            return impl

        # all other cases just set data
        # slice
        if isinstance(idx, types.SliceType):

            def impl_slice(S, idx, val):  # pragma: no cover
                bodo.hiframes.pd_series_ext.get_series_data(S)[idx] = (
                    bodo.utils.conversion.coerce_to_array(val, False)
                )

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
                raise BodoError(
                    "Indexing Series with Integer index using []"
                    " (which is label-based) not supported yet"
                )

            def impl_arr(S, idx, val):  # pragma: no cover
                idx_t = bodo.utils.conversion.coerce_to_array(idx)
                bodo.hiframes.pd_series_ext.get_series_data(S)[idx_t] = (
                    bodo.utils.conversion.coerce_to_array(val, False)
                )

            return impl_arr

        raise BodoError(f"Series [] setitem using {idx} not supported yet")


@overload(operator.setitem, no_unliteral=True)
def overload_array_list_setitem(A, idx, val):
    """Support setitem of Arrays with list/Series index (since not supported by Numba)"""
    if isinstance(A, types.Array) and isinstance(idx, (types.List, SeriesType)):

        def impl(A, idx, val):  # pragma: no cover
            A[bodo.utils.conversion.coerce_to_array(idx)] = val

        return impl


@overload(operator.getitem, no_unliteral=True)
def overload_const_index_series_getitem(S, idx):
    """handles label-based getitem on Series with constant Index values such as row
    input to df.apply() UDFs.
    """
    if (
        isinstance(S, (SeriesType, HeterogeneousSeriesType))
        and isinstance(S.index, HeterogeneousIndexType)
        and is_overload_constant_tuple(S.index.data)
    ):
        indices = get_overload_const_tuple(S.index.data)
        # Pandas falls back to positional indexing for int keys if index has no ints
        if isinstance(idx, types.Integer) and not any(
            isinstance(a, int) for a in indices
        ):
            return lambda S, idx: bodo.hiframes.pd_series_ext.get_series_data(S)[
                idx
            ]  # pragma: no cover

        # TODO(ehsan): support non-constant idx (rare but possible)
        if is_literal_type(idx):
            idx_val = get_literal_value(idx)
            if idx_val not in indices:  # pragma: no cover
                raise_bodo_error(
                    f"Series label-based getitem: '{idx_val}' not in {indices}"
                )

            arr_ind = indices.index(idx_val)

            return lambda S, idx: bodo.hiframes.pd_series_ext.get_series_data(S)[
                arr_ind
            ]  # pragma: no cover


@lower_cast(SeriesIatType, SeriesIatType)
@lower_cast(SeriesIlocType, SeriesIlocType)
@lower_cast(SeriesLocType, SeriesLocType)
def cast_series_iat(context, builder, fromty, toty, val):
    """cast indexing objects since 'dist' in Series/DataFrame data types can change
    in distributed analysis.
    See test_get_list_string
    """
    # just cast the underlying series/dataframe object
    iat_val = cgutils.create_struct_proxy(fromty)(context, builder, val)
    new_series = context.cast(builder, iat_val.obj, fromty.stype, toty.stype)
    new_iat_val = cgutils.create_struct_proxy(toty)(context, builder)
    new_iat_val.obj = new_series
    return new_iat_val._getvalue()


class HeterogeneousSeriesIlocType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = f"HeterogeneousSeriesIlocType({stype})"
        super().__init__(name)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    ndim = 1


@register_model(HeterogeneousSeriesIlocType)
class HeterogeneousSeriesIlocModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.stype)]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(HeterogeneousSeriesIlocType, "obj", "_obj")


@intrinsic
def init_heter_series_iloc(typingctx, obj):
    def codegen(context, builder, signature, args):
        (obj_val,) = args
        iloc_type = signature.return_type

        iloc_val = cgutils.create_struct_proxy(iloc_type)(context, builder)
        iloc_val.obj = obj_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], obj_val)

        return iloc_val._getvalue()

    return HeterogeneousSeriesIlocType(obj)(obj), codegen


@overload_attribute(HeterogeneousSeriesType, "iloc")
def overload_heter_series_iloc(s):
    return lambda s: bodo.hiframes.series_indexing.init_heter_series_iloc(
        s
    )  # pragma: no cover


@overload(operator.getitem, no_unliteral=True)
def overload_heter_series_iloc_getitem(I, idx):
    if isinstance(I, HeterogeneousSeriesIlocType) and is_overload_constant_int(idx):
        arr_ind = get_overload_const_int(idx)

        # TODO: box timedelta64, datetime.datetime/timedelta
        return lambda I, idx: bodo.utils.conversion.box_if_dt64(
            bodo.hiframes.pd_series_ext.get_series_data(I._obj)[arr_ind]
        )  # pragma: no cover
