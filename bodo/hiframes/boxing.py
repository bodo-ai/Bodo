# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Boxing and unboxing support for DataFrame, Series, etc.
"""
import decimal
import pandas as pd
import numpy as np
import datetime
import warnings
import numba
import math
from numba.extending import (
    typeof_impl,
    unbox,
    register_model,
    models,
    NativeValue,
    box,
    intrinsic,
)
from numba.core import types, cgutils
from numba.np import numpy_support
from numba.core.typing import signature

import bodo
from bodo.hiframes.pd_dataframe_ext import (
    DataFrameType,
    construct_dataframe,
    DataFramePayloadType,
)
from bodo.hiframes.datetime_date_ext import datetime_date_type, datetime_date_array_type
from bodo.libs.str_ext import string_type
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.list_str_arr_ext import list_string_array_type
from bodo.libs.struct_arr_ext import StructArrayType, StructType
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.int_arr_ext import typeof_pd_int_dtype
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.hiframes.pd_categorical_ext import PDCategoricalDtype
from bodo.hiframes.pd_series_ext import SeriesType, _get_series_array_type
from bodo.hiframes.split_impl import (
    string_array_split_view_type,
    box_str_arr_split_view,
)
from bodo.utils.typing import BodoWarning, BodoError

from bodo.libs import hstr_ext
from llvmlite import ir as lir
import llvmlite.binding as ll

ll.add_symbol("array_size", hstr_ext.array_size)
ll.add_symbol("array_getptr1", hstr_ext.array_getptr1)


@typeof_impl.register(pd.DataFrame)
def typeof_pd_dataframe(val, c):
    # convert "columns" from Index/MultiIndex to a tuple
    col_names = tuple(val.columns.to_list())
    col_types = get_hiframes_dtypes(val)
    index_typ = numba.typeof(val.index)

    return DataFrameType(col_types, index_typ, col_names, True)


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

    # unbox "columns" as a tuple instead of Index (for simplicity of codegen in
    # bodo-generated dataframes)
    columns_obj = c.pyapi.object_getattr_string(val, "columns")
    columns_list_obj = c.pyapi.call_method(columns_obj, "to_list", ())
    tuple_class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(tuple))
    columns_tup_obj = c.pyapi.call_function_objargs(
        tuple_class_obj, (columns_list_obj,)
    )
    columns_tup = c.pyapi.to_native_value(
        numba.typeof(typ.columns), columns_tup_obj
    ).value
    c.pyapi.decref(columns_obj)
    c.pyapi.decref(columns_list_obj)
    c.pyapi.decref(tuple_class_obj)
    c.pyapi.decref(columns_tup_obj)

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
        c.context, c.builder, typ, data_tup, index_val, columns_tup, unboxed_tup, val
    )

    return NativeValue(dataframe_val)


def get_hiframes_dtypes(df):
    """get hiframe data types for a pandas dataframe
    """
    col_names = df.columns.tolist()
    # TODO: remove pd int dtype hack
    hi_typs = [
        _get_series_array_type(_infer_series_dtype(df[cname])) for cname in col_names
    ]
    return tuple(hi_typs)


def _infer_series_dtype(S):
    if S.dtype == np.dtype("O"):
        # XXX: assume empty series/column is string since it's the most common
        # TODO: checks for distributed case with list/datetime.date/...
        # e.g. one rank's data is empty but other ranks have other types
        # XXX assuming the whole column is strings if 1st val is string
        # TODO: handle NA as 1st value
        i = 0
        while i < len(S) and (
            S.iloc[i] is np.nan
            or S.iloc[i] is None
            or (isinstance(S.iloc[i], float) and math.isnan(S.iloc[i]))
        ):
            i += 1
        if i == len(S):
            # assume all NA object column is string
            warnings.warn(
                BodoWarning(
                    "Empty object array passed to Bodo, which causes ambiguity in typing. "
                    "This can cause errors in parallel execution."
                )
            )
            return string_type

        first_val = S.iloc[i]
        # NOTE: pandas may create array(array(str)) instead of array(list(str))
        # see test_io.py::test_pq_list_str
        if isinstance(first_val, (list, np.ndarray)):
            return _infer_series_list_dtype(S.values, S.name)
        elif isinstance(first_val, str):
            return string_type
        elif isinstance(first_val, bool):
            return types.bool_  # will become BooleanArray in Series and DF
        # struct array: series of dict where all keys are strings
        elif isinstance(first_val, dict) and all(
            isinstance(k, str) for k in first_val.keys()
        ):
            field_names = tuple(first_val.keys())
            data_types = tuple(numba.typeof(v) for v in first_val.values())
            return StructType(data_types, field_names)
        elif isinstance(S.values[i], datetime.date):
            # XXX: using .values to check date type since DatetimeIndex returns
            # Timestamp which is subtype of datetime.date
            return datetime_date_type
        if isinstance(first_val, decimal.Decimal):
            # NOTE: converting decimal.Decimal objects to 38/18, same as Spark
            return Decimal128Type(38, 18)
        else:
            raise BodoError(
                "object dtype infer: data type for column {} not supported".format(
                    S.name
                )
            )

    # nullable int dtype
    if isinstance(S.dtype, pd.core.arrays.integer._IntegerDtype):
        return typeof_pd_int_dtype(S.dtype, None)
    elif isinstance(S.dtype, pd.CategoricalDtype):
        return PDCategoricalDtype(S.dtype.categories.to_list())
    elif isinstance(S.dtype, pd.StringDtype):
        return string_type
    elif isinstance(S.dtype, pd.BooleanDtype):
        return types.bool_

    # regular numpy types
    try:
        return numpy_support.from_dtype(S.dtype)
    except NotImplementedError:
        raise ValueError(
            "np dtype infer: data type for column {} not supported".format(S.name)
        )


def _infer_series_list_dtype(A, name):
    for i in range(len(A)):
        first_val = A[i]
        if isinstance(first_val, float) and np.isnan(first_val) or first_val is None:
            continue
        if not isinstance(first_val, (list, np.ndarray)):
            raise ValueError("data type for column {} not supported".format(name))
        if len(first_val) > 0:
            # TODO: support more types
            # TODO: can nulls be inside the list?
            list_val = first_val[0]
            try:
                dtype = numba.typeof(list_val)
                return types.List(dtype)
            except:
                raise ValueError("data type for column {} not supported".format(name))
    # assuming array of all empty lists is string by default
    return types.List(string_type)


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
    columns = dataframe.columns
    columns_typ = numba.typeof(typ.columns)
    context.nrt.incref(builder, columns_typ, columns)
    columns_obj = pyapi.from_native_value(columns_typ, columns, c.env_manager)

    with c.builder.if_else(has_parent) as (use_parent, otherwise):
        with use_parent:
            pyapi.incref(obj)
        with otherwise:
            # df_obj = pd.DataFrame()
            mod_name = context.insert_const_string(c.builder.module, "pandas")
            class_obj = pyapi.import_module_noblock(mod_name)
            df_obj = pyapi.call_method(class_obj, "DataFrame", ())
            pyapi.decref(class_obj)
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
            # df[c] = boxed_arr
            c_ind_obj = pyapi.long_from_longlong(context.get_constant(types.int64, i))
            col_name_obj = c.pyapi.object_getitem(columns_obj, c_ind_obj)

            # NOTE: adding extra incref() since boxing could be called twice on
            # a dataframe and not having incref can cause crashes.
            # see test_csv_double_box.
            # TODO: Find the right solution to refcounting in @box functions
            context.nrt.incref(builder, arr_typ, arr)
            arr_obj = pyapi.from_native_value(arr_typ, arr, c.env_manager)
            df_obj = builder.load(res)
            pyapi.object_setitem(df_obj, col_name_obj, arr_obj)
            pyapi.decref(arr_obj)
            pyapi.decref(c_ind_obj)
            pyapi.decref(col_name_obj)

    # set index if doesn't have parent
    with builder.if_then(builder.not_(has_parent)):
        # set df.index
        context.nrt.incref(builder, typ.index, dataframe_payload.index)
        arr_obj = c.pyapi.from_native_value(
            typ.index, dataframe_payload.index, c.env_manager
        )
        df_obj = builder.load(res)
        pyapi.object_setattr_string(df_obj, "index", arr_obj)
        pyapi.decref(arr_obj)
        # set df columns again to fix potential multi-index issues
        # see test_dataframe.py::test_unbox_df_multi
        pyapi.object_setattr_string(df_obj, "columns", columns_obj)

    pyapi.decref(columns_obj)
    # decref() should be called on native value
    # see https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/boxing.py#L389
    c.context.nrt.decref(c.builder, typ, val)
    return c.builder.load(res)


@intrinsic
def unbox_dataframe_column(typingctx, df, i=None):
    def codegen(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        c = numba.core.pythonapi._UnboxContext(context, builder, pyapi)
        gil_state = pyapi.gil_ensure()  # acquire GIL

        df_typ = sig.args[0]
        col_ind = sig.args[1].literal_value
        data_typ = df_typ.data[col_ind]
        columns_typ = numba.typeof(df_typ.columns)
        # TODO: refcounts?

        dataframe = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0]
        )
        columns_obj = pyapi.from_native_value(
            columns_typ, dataframe.columns, context.get_env_manager(builder)
        )
        # XXX: this incref seems necessary, which is probably due to Numba's limitations
        # in boxing of tuples
        # TODO: fix boxing refcount
        context.nrt.incref(builder, columns_typ, dataframe.columns)
        # no decref for 'col_name_obj' since tuple_getitem returns borrowed reference
        col_name_obj = pyapi.tuple_getitem(columns_obj, col_ind)
        series_obj = c.pyapi.object_getitem(dataframe.parent, col_name_obj)
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
        ptrty = context.get_data_type(payload_type).as_pointer()
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


@box(SeriesType)
def box_series(typ, val, c):
    """
    """
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    dtype = typ.dtype

    # TODO: handle parent
    series_payload = bodo.hiframes.pd_series_ext.get_series_payload(
        c.context, c.builder, typ, val
    )

    # box data/index/name
    # incref since boxing functions steal a reference
    c.context.nrt.incref(c.builder, typ.data, series_payload.data)
    c.context.nrt.incref(c.builder, typ.index, series_payload.index)
    c.context.nrt.incref(c.builder, typ.name_typ, series_payload.name)
    arr_obj = _box_series_data(dtype, typ.data, series_payload.data, c)
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


def _box_series_data(dtype, data_typ, val, c):

    if isinstance(dtype, types.BaseTuple):
        np_dtype = np.dtype(",".join(str(t) for t in dtype.types), align=True)
        dtype = numpy_support.from_dtype(np_dtype)

    arr = c.pyapi.from_native_value(data_typ, val, c.env_manager)

    if isinstance(dtype, types.Record):
        o_str = c.context.insert_const_string(c.builder.module, "O")
        o_str = c.pyapi.string_from_string(o_str)
        arr = c.pyapi.call_method(arr, "astype", (o_str,))

    return arr


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
        dtype = _infer_ndarray_obj_dtype(val)
        if dtype == string_type:
            return string_array_type
        if dtype == types.bool_:
            return bodo.libs.bool_arr_ext.boolean_array
        if dtype == types.List(string_type):
            return list_string_array_type
        if isinstance(dtype, types.List):
            return ArrayItemArrayType(dtype.dtype)
        if dtype == datetime_date_type:
            return datetime_date_array_type  # TODO: test array of datetime.date
        if isinstance(dtype, Decimal128Type):
            return DecimalArrayType(dtype.precision, dtype.scale)
        if isinstance(dtype, StructType):
            data = tuple(
                bodo.hiframes.pd_series_ext._get_series_array_type(t)
                for t in dtype.data
            )
            return StructArrayType(data, dtype.names)
        raise BodoError(
            "Unsupported array dtype: {}".format(val.dtype)
        )  # pragma: no cover

    layout = numba.np.numpy_support.map_layout(val)
    readonly = not val.flags.writeable
    return types.Array(dtype, val.ndim, layout, readonly=readonly)


def _infer_ndarray_obj_dtype(val):
    # strings only have object dtype, TODO: support fixed size np strings
    if not val.dtype == np.dtype("O"):  # pragma: no cover
        raise BodoError("Unsupported array dtype: {}".format(val.dtype))

    # XXX assuming the whole array is strings if 1st val is string
    i = 0
    # is_scalar call necessary since pd.isna() treats list of string as array
    while i < len(val) and pd.api.types.is_scalar(val[i]) and pd.isna(val[i]):
        i += 1
    if i == len(val):
        # empty or all NA object arrays are assumed to be strings
        warnings.warn(
            BodoWarning(
                "Empty object array passed to Bodo, which causes ambiguity in typing. "
                "This can cause errors in parallel execution."
            )
        )
        return string_type

    first_val = val[i]
    if isinstance(first_val, str):
        return string_type
    elif isinstance(first_val, bool):
        return types.bool_
    # assuming object arrays with dictionary values are struct arrays, which means all
    # keys are string and match across dictionaries, and all values with same key have
    # same data type
    elif isinstance(first_val, dict) and all(
        isinstance(k, str) for k in first_val.keys()
    ):
        field_names = tuple(first_val.keys())
        data_types = tuple(numba.typeof(v) for v in first_val.values())
        return StructType(data_types, field_names)
    if isinstance(first_val, (list, np.ndarray)):
        return bodo.hiframes.boxing._infer_series_list_dtype(val, "array")
    if isinstance(first_val, datetime.date):
        return datetime_date_type
    if isinstance(first_val, decimal.Decimal):
        # NOTE: converting decimal.Decimal objects to 38/18, same as Spark
        return Decimal128Type(38, 18)

    raise BodoError(
        "Unsupported object array with first value: {}".format(first_val)
    )  # pragma: no cover


# TODO: support array of strings
# @typeof_impl.register(np.ndarray)
# def typeof_np_string(val, c):
#     arr_typ = numba.core.typing.typeof._typeof_ndarray(val, c)
#     # match string dtype
#     if isinstance(arr_typ.dtype, (types.UnicodeCharSeq, types.CharSeq)):
#         return string_array_type
#     return arr_typ
