# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Boxing and unboxing support for DataFrame, Series, etc.
"""
import datetime
import decimal
import warnings

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.typing import signature
from numba.extending import NativeValue, box, intrinsic, typeof_impl, unbox
from numba.np import numpy_support
from numba.typed.typeddict import Dict

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.datetime_timedelta_ext import datetime_timedelta_array_type
from bodo.hiframes.pd_dataframe_ext import (
    DataFramePayloadType,
    DataFrameType,
    construct_dataframe,
)
from bodo.hiframes.pd_series_ext import HeterogeneousSeriesType, SeriesType
from bodo.hiframes.split_impl import string_array_split_view_type
from bodo.libs import hstr_ext
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.int_arr_ext import typeof_pd_int_dtype
from bodo.libs.map_arr_ext import MapArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    dtype_to_array_type,
    get_overload_const_int,
    is_overload_constant_int,
    to_nullable_type,
)

ll.add_symbol("array_size", hstr_ext.array_size)
ll.add_symbol("array_getptr1", hstr_ext.array_getptr1)


@typeof_impl.register(pd.DataFrame)
def typeof_pd_dataframe(val, c):
    # convert "columns" from Index/MultiIndex to a tuple
    col_names = tuple(val.columns.to_list())
    col_types = get_hiframes_dtypes(val)
    index_typ = numba.typeof(val.index)

    return DataFrameType(col_types, index_typ, col_names)


# register series types for import
@typeof_impl.register(pd.Series)
def typeof_pd_series(val, c):
    return SeriesType(
        _infer_series_dtype(val),
        index=numba.typeof(val.index),
        name_typ=numba.typeof(val.name),
    )


@unbox(DataFrameType)
def unbox_dataframe(typ, val, c):
    """unbox dataframe to an empty DataFrame struct
    columns will be extracted later if necessary.
    """
    n_cols = len(typ.columns)

    # set all columns as not unboxed
    zero = c.context.get_constant(types.int8, 0)
    unboxed_tup = c.context.make_tuple(
        c.builder, types.UniTuple(types.int8, n_cols + 1), [zero] * (n_cols + 1)
    )

    # unbox index
    # TODO: unbox index only if necessary
    ind_obj = c.pyapi.object_getattr_string(val, "index")
    index_val = c.pyapi.to_native_value(typ.index, ind_obj).value
    c.pyapi.decref(ind_obj)

    # set data arrays as null due to lazy unboxing
    # TODO: does this work for array types?
    data_nulls = [c.context.get_constant_null(t) for t in typ.data]
    data_tup = c.context.make_tuple(c.builder, types.Tuple(typ.data), data_nulls)

    dataframe_val = construct_dataframe(
        c.context, c.builder, typ, data_tup, index_val, unboxed_tup, val
    )

    return NativeValue(dataframe_val)


def get_hiframes_dtypes(df):
    """get hiframe data types for a pandas dataframe"""
    hi_typs = [
        dtype_to_array_type(_infer_series_dtype(df.iloc[:, i]))
        for i in range(len(df.columns))
    ]
    return tuple(hi_typs)


def _infer_series_dtype(S):
    if S.dtype == np.dtype("O"):
        return numba.typeof(S.values).dtype

    # nullable int dtype
    if isinstance(S.dtype, pd.core.arrays.integer._IntegerDtype):
        return typeof_pd_int_dtype(S.dtype, None)
    elif isinstance(S.dtype, pd.CategoricalDtype):
        return bodo.typeof(S.dtype)
    elif isinstance(S.dtype, pd.StringDtype):
        return string_type
    elif isinstance(S.dtype, pd.BooleanDtype):
        return types.bool_

    if isinstance(S.dtype, pd.DatetimeTZDtype):
        raise BodoError("Timezone-aware datetime data type not supported yet")

    # regular numpy types
    try:
        return numpy_support.from_dtype(S.dtype)
    except:  # pragma: no cover
        raise BodoError(f"data type {S.dtype} for column {S.name} not supported yet")


@box(DataFrameType)
def box_dataframe(typ, val, c):
    """Boxes native dataframe value into Python dataframe object, required for function
    return, printing, object mode, etc.
    Works by boxing individual data arrays.
    """
    context = c.context
    builder = c.builder
    pyapi = c.pyapi

    dataframe_payload = bodo.hiframes.pd_dataframe_ext.get_dataframe_payload(
        c.context, c.builder, typ, val
    )
    dataframe = cgutils.create_struct_proxy(typ)(context, builder, value=val)

    # see boxing of reflected list in Numba:
    # https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/boxing.py#L561
    obj = dataframe.parent
    res = cgutils.alloca_once_value(c.builder, obj)
    # df unboxed from Python
    has_parent = cgutils.is_not_null(builder, obj)

    # get column names object
    # TODO: avoid generating large tuples and lower a constant Index if possible
    # (e.g. if homogeneous names)
    columns_typ = numba.typeof(typ.columns)
    columns = context.get_constant_generic(builder, columns_typ, typ.columns)
    context.nrt.incref(builder, columns_typ, columns)
    columns_obj = pyapi.from_native_value(columns_typ, columns, c.env_manager)

    with c.builder.if_else(has_parent) as (use_parent, otherwise):
        with use_parent:
            pyapi.incref(obj)
            # set parent dataframe column names to numbers for robust setting of columns
            # df.columns = np.arange(len(df.columns))
            mod_name = context.insert_const_string(c.builder.module, "numpy")
            class_obj = pyapi.import_module_noblock(mod_name)
            n_cols_obj = pyapi.long_from_longlong(
                lir.Constant(lir.IntType(64), len(typ.columns))
            )
            col_nums_arr_obj = pyapi.call_method(class_obj, "arange", (n_cols_obj,))
            pyapi.object_setattr_string(obj, "columns", col_nums_arr_obj)
            pyapi.decref(class_obj)
            pyapi.decref(col_nums_arr_obj)
            pyapi.decref(n_cols_obj)
        with otherwise:
            # df_obj = pd.DataFrame(index=index)
            context.nrt.incref(builder, typ.index, dataframe_payload.index)
            index_obj = c.pyapi.from_native_value(
                typ.index, dataframe_payload.index, c.env_manager
            )

            mod_name = context.insert_const_string(c.builder.module, "pandas")
            class_obj = pyapi.import_module_noblock(mod_name)
            df_obj = pyapi.call_method(
                class_obj, "DataFrame", (pyapi.borrow_none(), index_obj)
            )
            pyapi.decref(class_obj)
            pyapi.decref(index_obj)
            builder.store(df_obj, res)

    # get data arrays and box them
    n_cols = len(typ.columns)
    col_arrs = [builder.extract_value(dataframe_payload.data, i) for i in range(n_cols)]
    arr_typs = typ.data
    for i, arr, arr_typ in zip(range(n_cols), col_arrs, arr_typs):
        # box array if df doesn't have a parent, or column was unboxed in function,
        # since changes in arrays like strings don't reflect back to parent object
        unboxed = builder.extract_value(dataframe_payload.unboxed, i)
        is_unboxed = builder.icmp_unsigned("==", unboxed, lir.Constant(unboxed.type, 1))
        box_array = builder.or_(
            builder.not_(has_parent), builder.and_(has_parent, is_unboxed)
        )

        with builder.if_then(box_array):
            # df[i] = boxed_arr
            c_ind_obj = pyapi.long_from_longlong(context.get_constant(types.int64, i))

            context.nrt.incref(builder, arr_typ, arr)
            arr_obj = pyapi.from_native_value(arr_typ, arr, c.env_manager)
            df_obj = builder.load(res)
            pyapi.object_setitem(df_obj, c_ind_obj, arr_obj)
            pyapi.decref(arr_obj)
            pyapi.decref(c_ind_obj)

    df_obj = builder.load(res)
    # set df columns separately to support repeated names and fix potential multi-index
    # issues, see test_dataframe.py::test_unbox_df_multi, test_box_repeated_names
    pyapi.object_setattr_string(df_obj, "columns", columns_obj)
    pyapi.decref(columns_obj)
    # decref() should be called on native value
    # see https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/boxing.py#L389
    c.context.nrt.decref(c.builder, typ, val)
    return df_obj


@intrinsic
def unbox_dataframe_column(typingctx, df, i=None):
    assert isinstance(df, DataFrameType) and is_overload_constant_int(i)

    def codegen(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        c = numba.core.pythonapi._UnboxContext(context, builder, pyapi)
        gil_state = pyapi.gil_ensure()  # acquire GIL

        df_typ = sig.args[0]
        col_ind = get_overload_const_int(sig.args[1])
        data_typ = df_typ.data[col_ind]
        # TODO: refcounts?

        dataframe = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0]
        )
        # generate df.iloc[:,i] for parent dataframe object
        none_obj = c.pyapi.borrow_none()
        slice_class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(slice))
        slice_obj = c.pyapi.call_function_objargs(slice_class_obj, [none_obj])
        col_ind_obj = c.pyapi.long_from_longlong(args[1])
        slice_ind_tup_obj = c.pyapi.tuple_pack([slice_obj, col_ind_obj])

        df_iloc_obj = c.pyapi.object_getattr_string(dataframe.parent, "iloc")
        series_obj = c.pyapi.object_getitem(df_iloc_obj, slice_ind_tup_obj)
        arr_obj_orig = c.pyapi.object_getattr_string(series_obj, "values")

        if isinstance(data_typ, types.Array):
            # call np.ascontiguousarray() on array since it may not be contiguous
            # the typing infrastructure assumes C-contiguous arrays
            # see test_df_multi_get_level() for an example of non-contiguous input
            np_mod_name = c.context.insert_const_string(c.builder.module, "numpy")
            np_class_obj = c.pyapi.import_module_noblock(np_mod_name)
            arr_obj = c.pyapi.call_method(
                np_class_obj, "ascontiguousarray", (arr_obj_orig,)
            )
            c.pyapi.decref(arr_obj_orig)
            c.pyapi.decref(np_class_obj)
        else:
            arr_obj = arr_obj_orig

        # TODO: support column of tuples?
        native_val = _unbox_series_data(data_typ.dtype, data_typ, arr_obj, c)

        c.pyapi.decref(slice_class_obj)
        c.pyapi.decref(slice_obj)
        c.pyapi.decref(col_ind_obj)
        c.pyapi.decref(slice_ind_tup_obj)
        c.pyapi.decref(df_iloc_obj)
        c.pyapi.decref(series_obj)
        c.pyapi.decref(arr_obj)
        pyapi.gil_release(gil_state)  # release GIL

        # assign array and set unboxed flag
        dataframe_payload = bodo.hiframes.pd_dataframe_ext.get_dataframe_payload(
            c.context, c.builder, df_typ, args[0]
        )
        dataframe_payload.data = builder.insert_value(
            dataframe_payload.data, native_val.value, col_ind
        )
        dataframe_payload.unboxed = builder.insert_value(
            dataframe_payload.unboxed, context.get_constant(types.int8, 1), col_ind
        )

        # store payload
        payload_type = DataFramePayloadType(df_typ)
        payload_ptr = context.nrt.meminfo_data(builder, dataframe.meminfo)
        ptrty = context.get_value_type(payload_type).as_pointer()
        payload_ptr = builder.bitcast(payload_ptr, ptrty)
        builder.store(dataframe_payload._getvalue(), payload_ptr)

    return signature(types.none, df, i), codegen


@unbox(SeriesType)
def unbox_series(typ, val, c):
    arr_obj_orig = c.pyapi.object_getattr_string(val, "values")

    if isinstance(typ.data, types.Array):
        # make contiguous by calling np.ascontiguousarray()
        np_mod_name = c.context.insert_const_string(c.builder.module, "numpy")
        np_class_obj = c.pyapi.import_module_noblock(np_mod_name)
        arr_obj = c.pyapi.call_method(
            np_class_obj, "ascontiguousarray", (arr_obj_orig,)
        )
        c.pyapi.decref(arr_obj_orig)
        c.pyapi.decref(np_class_obj)
    else:
        arr_obj = arr_obj_orig

    data_val = _unbox_series_data(typ.dtype, typ.data, arr_obj, c).value

    index_obj = c.pyapi.object_getattr_string(val, "index")
    index_val = c.pyapi.to_native_value(typ.index, index_obj).value

    name_obj = c.pyapi.object_getattr_string(val, "name")
    name_val = c.pyapi.to_native_value(typ.name_typ, name_obj).value

    series_val = bodo.hiframes.pd_series_ext.construct_series(
        c.context, c.builder, typ, data_val, index_val, name_val
    )
    # TODO: set parent pointer
    c.pyapi.decref(arr_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(name_obj)
    return NativeValue(series_val)


def _unbox_series_data(dtype, data_typ, arr_obj, c):
    if data_typ == string_array_split_view_type:
        # XXX dummy unboxing to avoid errors in _get_dataframe_data()
        out_view = c.context.make_helper(c.builder, string_array_split_view_type)
        return NativeValue(out_view._getvalue())

    return c.pyapi.to_native_value(data_typ, arr_obj)


@box(HeterogeneousSeriesType)
@box(SeriesType)
def box_series(typ, val, c):
    """"""
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    # TODO: handle parent
    series_payload = bodo.hiframes.pd_series_ext.get_series_payload(
        c.context, c.builder, typ, val
    )

    # box data/index/name
    # incref since boxing functions steal a reference
    c.context.nrt.incref(c.builder, typ.data, series_payload.data)
    c.context.nrt.incref(c.builder, typ.index, series_payload.index)
    c.context.nrt.incref(c.builder, typ.name_typ, series_payload.name)
    arr_obj = c.pyapi.from_native_value(typ.data, series_payload.data, c.env_manager)
    index_obj = c.pyapi.from_native_value(
        typ.index, series_payload.index, c.env_manager
    )
    name_obj = c.pyapi.from_native_value(
        typ.name_typ, series_payload.name, c.env_manager
    )

    # call pd.Series()
    dtype = c.pyapi.make_none()  # TODO: dtype
    res = c.pyapi.call_method(
        pd_class_obj, "Series", (arr_obj, index_obj, dtype, name_obj)
    )
    c.pyapi.decref(arr_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(name_obj)

    c.pyapi.decref(pd_class_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return res


# --------------- typeof support for object arrays --------------------


# XXX: this is overwriting Numba's array type registration, make sure it is
# robust
# TODO: support other array types like datetime.date
@typeof_impl.register(np.ndarray)
def _typeof_ndarray(val, c):
    try:
        dtype = numba.np.numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        dtype = types.pyobject

    if dtype == types.pyobject:
        return _infer_ndarray_obj_dtype(val)

    layout = numba.np.numpy_support.map_layout(val)
    readonly = not val.flags.writeable
    return types.Array(dtype, val.ndim, layout, readonly=readonly)


def _infer_ndarray_obj_dtype(val):
    # strings only have object dtype, TODO: support fixed size np strings
    if not val.dtype == np.dtype("O"):  # pragma: no cover
        raise BodoError("Unsupported array dtype: {}".format(val.dtype))

    # XXX assuming the whole array is strings if 1st val is string
    i = 0
    # skip NAs and empty lists/arrays (for array(item) array cases)
    # is_scalar call necessary since pd.isna() treats list of string as array
    while i < len(val) and (
        (pd.api.types.is_scalar(val[i]) and pd.isna(val[i]))
        or (not pd.api.types.is_scalar(val[i]) and len(val[i]) == 0)
    ):
        i += 1
    if i == len(val):
        # empty or all NA object arrays are assumed to be strings
        warnings.warn(
            BodoWarning(
                "Empty object array passed to Bodo, which causes ambiguity in typing. "
                "This can cause errors in parallel execution."
            )
        )
        return string_array_type

    first_val = val[i]
    if isinstance(first_val, str):
        return string_array_type
    elif isinstance(first_val, bool):
        return bodo.libs.bool_arr_ext.boolean_array
    elif isinstance(first_val, (int, np.int32, np.int64)):
        return bodo.libs.int_arr_ext.IntegerArrayType(numba.typeof(first_val))
    # assuming object arrays with dictionary values string keys are struct arrays, which
    # means all keys are string and match across dictionaries, and all values with same
    # key have same data type
    # TODO: distinguish between Struct and Map arrays properly
    elif isinstance(first_val, (dict, Dict)) and all(
        isinstance(k, str) for k in first_val.keys()
    ):
        field_names = tuple(first_val.keys())
        # TODO: handle None value in first_val elements
        data_types = tuple(_get_struct_value_arr_type(v) for v in first_val.values())
        return StructArrayType(data_types, field_names)
    elif isinstance(first_val, (dict, Dict)):
        key_arr_type = numba.typeof(_value_to_array(list(first_val.keys())))
        value_arr_type = numba.typeof(_value_to_array(list(first_val.values())))
        # TODO: handle 2D ndarray case
        return MapArrayType(key_arr_type, value_arr_type)
    elif isinstance(first_val, tuple):
        data_types = tuple(_get_struct_value_arr_type(v) for v in first_val)
        return TupleArrayType(data_types)
    if isinstance(
        first_val,
        (
            list,
            np.ndarray,
            pd.arrays.BooleanArray,
            pd.arrays.IntegerArray,
            pd.arrays.StringArray,
        ),
    ):
        # normalize list to array, 'np.object_' dtype to consider potential nulls
        if isinstance(first_val, list):
            first_val = _value_to_array(first_val)
        val_typ = numba.typeof(first_val)
        return ArrayItemArrayType(val_typ)
    if isinstance(first_val, datetime.date):
        return datetime_date_array_type
    if isinstance(first_val, datetime.timedelta):
        return datetime_timedelta_array_type
    if isinstance(first_val, decimal.Decimal):
        # NOTE: converting decimal.Decimal objects to 38/18, same as Spark
        return DecimalArrayType(38, 18)

    raise BodoError(
        "Unsupported object array with first value: {}".format(first_val)
    )  # pragma: no cover


def _value_to_array(val):
    """convert list or dict value to object array for typing purposes"""
    assert isinstance(val, (list, dict, Dict))
    if isinstance(val, (dict, Dict)):
        val = dict(val)
        return np.array([val], np.object_)

    # add None to list to avoid Numpy's automatic conversion to 2D arrays
    val_infer = val.copy()
    val_infer.append(None)
    arr = np.array(val_infer, np.object_)

    # assume float lists can be regular np.float64 arrays
    # TODO handle corener cases where None could be used as NA instead of np.nan
    if len(val) and isinstance(val[0], float):
        arr = np.array(val, np.float64)
    return arr


def _get_struct_value_arr_type(v):
    """get data array type for a field value of a struct array"""
    if isinstance(v, (dict, Dict)):
        return numba.typeof(_value_to_array(v))

    if isinstance(v, list):
        return dtype_to_array_type(numba.typeof(_value_to_array(v)))

    if pd.api.types.is_scalar(v) and pd.isna(v):
        # assume string array if first field value is NA
        # TODO: use other rows until non-NA is found
        warnings.warn(
            BodoWarning(
                "Field value in struct array is NA, which causes ambiguity in typing. "
                "This can cause errors in parallel execution."
            )
        )
        return string_array_type

    arr_typ = dtype_to_array_type(numba.typeof(v))
    # use nullable arrays for integer/bool values since there could be None objects
    if isinstance(v, (int, bool)):
        arr_typ = to_nullable_type(arr_typ)

    return arr_typ


# TODO: support array of strings
# @typeof_impl.register(np.ndarray)
# def typeof_np_string(val, c):
#     arr_typ = numba.core.typing.typeof._typeof_ndarray(val, c)
#     # match string dtype
#     if isinstance(arr_typ.dtype, (types.UnicodeCharSeq, types.CharSeq)):
#         return string_array_type
#     return arr_typ
