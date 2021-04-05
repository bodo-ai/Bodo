# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implement pd.Series typing and data model handling.
"""
import operator

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed
from numba.core.typing.templates import (
    AttributeTemplate,
    bound_function,
    signature,
)
from numba.extending import (
    infer_getattr,
    intrinsic,
    lower_builtin,
    lower_cast,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
)

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.hiframes.datetime_timedelta_ext import pd_timedelta_type
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArrayType,
    PDCategoricalDtype,
)
from bodo.hiframes.pd_timestamp_ext import pd_timestamp_type
from bodo.io import csv_cpp
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.libs.int_arr_ext import IntDtype, IntegerArrayType
from bodo.libs.map_arr_ext import MapArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type, unicode_to_utf8
from bodo.libs.struct_arr_ext import StructArrayType, StructType
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.utils.transform import get_const_func_output_type
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    create_unsupported_overload,
    get_overload_const_tuple,
    get_udf_error_msg,
    get_udf_out_arr_type,
    is_heterogeneous_tuple_type,
    is_overload_constant_str,
    is_overload_constant_tuple,
    is_overload_false,
    is_overload_none,
)

_csv_output_is_dir = types.ExternalFunction(
    "csv_output_is_dir",
    types.int8(types.voidptr),
)
ll.add_symbol("csv_output_is_dir", csv_cpp.csv_output_is_dir)


class SeriesType(types.IterableType, types.ArrayCompatible):
    """Type class for Series objects"""

    ndim = 1

    def __init__(self, dtype, data=None, index=None, name_typ=None):
        from bodo.hiframes.pd_index_ext import RangeIndexType

        # keeping data array in type since operators can make changes such
        # as making array unaligned etc.
        # data is underlying array type and can have different dtype
        data = _get_series_array_type(dtype) if data is None else data
        # store regular dtype instead of IntDtype to avoid errors
        dtype = dtype.dtype if isinstance(dtype, IntDtype) else dtype
        self.dtype = dtype
        self.data = data
        name_typ = types.none if name_typ is None else name_typ
        index = RangeIndexType(types.none) if index is None else index
        self.index = index  # index should be an Index type (not Array)
        self.name_typ = name_typ
        super(SeriesType, self).__init__(
            name="series({}, {}, {}, {})".format(dtype, data, index, name_typ)
        )

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self, dtype=None, index=None):
        # XXX is copy necessary?
        if index is None:
            index = self.index.copy()
        dtype = dtype if dtype is not None else self.dtype
        data = _get_series_array_type(dtype)
        return SeriesType(dtype, data, index, self.name_typ)

    @property
    def key(self):
        # needed?
        return self.dtype, self.data, self.index, self.name_typ

    def unify(self, typingctx, other):
        if isinstance(other, SeriesType):
            new_index = self.index.unify(typingctx, other.index)

            # If dtype matches or other.dtype is undefined (inferred)
            if other.dtype == self.dtype or not other.dtype.is_precise():
                return SeriesType(
                    self.dtype, self.data.unify(typingctx, other.data), new_index
                )

        # XXX: unify Series/Array as Array
        return super(SeriesType, self).unify(typingctx, other)

    # XXX too dangerous, is it needed?
    # def can_convert_to(self, typingctx, other):
    #     # same as types.Array
    #     if (isinstance(other, SeriesType) and other.dtype == self.dtype):
    #         # called for overload selection sometimes
    #         # TODO: fix index
    #         if self.index == types.none and other.index == types.none:
    #             return self.data.can_convert_to(typingctx, other.data)
    #         if self.index != types.none and other.index != types.none:
    #             return max(self.data.can_convert_to(typingctx, other.data),
    #                 self.index.can_convert_to(typingctx, other.index))

    def is_precise(self):
        # same as types.Array
        return self.dtype.is_precise()

    @property
    def iterator_type(self):
        # same as Buffer
        # TODO: fix timestamp
        return self.data.iterator_type


class HeterogeneousSeriesType(types.Type):
    """
    Type class for Series objects with heterogeneous values (e.g. pd.Series([1, 'A']))
    """

    ndim = 1

    def __init__(self, data=None, index=None, name_typ=None):
        from bodo.hiframes.pd_index_ext import RangeIndexType

        self.data = data
        name_typ = types.none if name_typ is None else name_typ
        index = RangeIndexType(types.none) if index is None else index
        # TODO(ehsan): add check for index type
        self.index = index  # index should be an Index type (not Array)
        self.name_typ = name_typ
        super(HeterogeneousSeriesType, self).__init__(
            name="heter_series({}, {}, {})".format(data, index, name_typ)
        )

    def copy(self, index=None):
        if index is None:
            index = self.index.copy()
        return HeterogeneousSeriesType(self.data, index, self.name_typ)

    @property
    def key(self):
        return self.data, self.index, self.name_typ


@lower_builtin("getiter", SeriesType)
def series_getiter(context, builder, sig, args):
    """support getting an iterator object for Series by calling 'getiter' on the
    underlying array.
    """
    series_payload = get_series_payload(context, builder, sig.args[0], args[0])
    impl = context.get_function("getiter", sig.return_type(sig.args[0].data))
    return impl(builder, (series_payload.data,))


@infer_getattr
class HeterSeriesAttribute(AttributeTemplate):
    key = HeterogeneousSeriesType

    def generic_resolve(self, S, attr):
        """Handle getattr on row Series values pass to df.apply() UDFs."""
        from bodo.hiframes.pd_index_ext import HeterogeneousIndexType

        if isinstance(S.index, HeterogeneousIndexType) and is_overload_constant_tuple(
            S.index.data
        ):
            indices = get_overload_const_tuple(S.index.data)
            if attr in indices:
                arr_ind = indices.index(attr)
                return S.data[arr_ind]


def _get_series_array_type(dtype):
    """get underlying default array type of series based on its dtype"""
    dtype = types.unliteral(dtype)

    # UDFs may return lists, but we store array of array for output
    if isinstance(dtype, types.List):
        dtype = _get_series_array_type(dtype.dtype)

    # string array
    if dtype == string_type:
        return string_array_type

    if bodo.utils.utils.is_array_typ(dtype, False):
        return ArrayItemArrayType(dtype)

    # categorical
    if isinstance(dtype, PDCategoricalDtype):
        return CategoricalArrayType(dtype)

    if isinstance(dtype, IntDtype):
        return IntegerArrayType(dtype.dtype)

    if dtype == types.bool_:
        return boolean_array

    if dtype == datetime_date_type:
        return bodo.hiframes.datetime_date_ext.datetime_date_array_type

    if isinstance(dtype, Decimal128Type):
        return DecimalArrayType(dtype.precision, dtype.scale)

    if isinstance(dtype, StructType):
        return StructArrayType(
            tuple(_get_series_array_type(t) for t in dtype.data), dtype.names
        )

    if isinstance(dtype, types.BaseTuple):
        return TupleArrayType(tuple(_get_series_array_type(t) for t in dtype.types))

    if isinstance(dtype, types.DictType):
        return MapArrayType(
            _get_series_array_type(dtype.key_type),
            _get_series_array_type(dtype.value_type),
        )

    # PandasTimestamp becomes dt64 array
    if dtype == bodo.pd_timestamp_type:
        return types.Array(bodo.datetime64ns, 1, "C")

    if dtype == bodo.pd_timedelta_type:
        return types.Array(bodo.timedelta64ns, 1, "C")

    # TODO: other types?
    # regular numpy array
    return types.Array(dtype, 1, "C")


def is_str_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == string_type


def is_dt64_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == types.NPDatetime("ns")


def is_timedelta64_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == types.NPTimedelta("ns")


def is_datetime_date_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == datetime_date_type


# payload type inside meminfo so that mutation are seen by all references
class SeriesPayloadType(types.Type):
    def __init__(self, series_type):
        self.series_type = series_type
        super(SeriesPayloadType, self).__init__(
            name="SeriesPayloadType({})".format(series_type)
        )


@register_model(SeriesPayloadType)
class SeriesPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.series_type.data),
            ("index", fe_type.series_type.index),
            ("name", fe_type.series_type.name_typ),
        ]
        super(SeriesPayloadModel, self).__init__(dmm, fe_type, members)


@register_model(HeterogeneousSeriesType)
@register_model(SeriesType)
class SeriesModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        payload_type = SeriesPayloadType(fe_type)
        # payload_type = types.Opaque('Opaque.Series')
        # TODO: does meminfo decref content when object is deallocated?
        members = [
            ("meminfo", types.MemInfoPointer(payload_type)),
            # for boxed Series, enables updating original Series object
            ("parent", types.pyobject),
        ]
        super(SeriesModel, self).__init__(dmm, fe_type, members)


def define_series_dtor(context, builder, series_type, payload_type):
    """
    Define destructor for Series type if not already defined
    """
    mod = builder.module
    # Declare dtor
    fnty = lir.FunctionType(lir.VoidType(), [cgutils.voidptr_t])
    # TODO(ehsan): do we need to sanitize the name in any case?
    fn = mod.get_or_insert_function(fnty, name=".dtor.series.{}".format(series_type))

    # End early if the dtor is already defined
    if not fn.is_declaration:
        return fn

    fn.linkage = "linkonce_odr"
    # Populate the dtor
    builder = lir.IRBuilder(fn.append_basic_block())
    base_ptr = fn.args[0]  # void*

    # get payload struct
    ptrty = context.get_value_type(payload_type).as_pointer()
    payload_ptr = builder.bitcast(base_ptr, ptrty)
    payload = context.make_helper(builder, payload_type, ref=payload_ptr)

    context.nrt.decref(builder, series_type.data, payload.data)
    context.nrt.decref(builder, series_type.index, payload.index)
    context.nrt.decref(builder, series_type.name_typ, payload.name)

    builder.ret_void()
    return fn


def construct_series(context, builder, series_type, data_val, index_val, name_val):
    # create payload struct and store values
    payload_type = SeriesPayloadType(series_type)
    series_payload = cgutils.create_struct_proxy(payload_type)(context, builder)
    series_payload.data = data_val
    series_payload.index = index_val
    series_payload.name = name_val

    # create meminfo and store payload
    payload_ll_type = context.get_value_type(payload_type)
    payload_size = context.get_abi_sizeof(payload_ll_type)
    dtor_fn = define_series_dtor(context, builder, series_type, payload_type)
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, payload_size), dtor_fn
    )
    meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, payload_ll_type.as_pointer())
    builder.store(series_payload._getvalue(), meminfo_data_ptr)

    # create Series struct
    series = cgutils.create_struct_proxy(series_type)(context, builder)
    series.meminfo = meminfo
    # Set parent to NULL
    series.parent = cgutils.get_null_value(series.parent.type)
    return series._getvalue()


@intrinsic
def init_series(typingctx, data, index, name=None):
    """Create a Series with provided data, index and name values.
    Used as a single constructor for Series and assigning its data, so that
    optimization passes can look for init_series() to see if underlying
    data has changed, and get the array variables from init_series() args if
    not changed.
    """
    from bodo.hiframes.pd_index_ext import is_pd_index_type
    from bodo.hiframes.pd_multi_index_ext import MultiIndexType

    assert is_pd_index_type(index) or isinstance(index, MultiIndexType)
    name = types.none if name is None else name

    def codegen(context, builder, signature, args):
        data_val, index_val, name_val = args
        series_type = signature.return_type

        series_val = construct_series(
            context, builder, series_type, data_val, index_val, name_val
        )

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_val)
        context.nrt.incref(builder, signature.args[1], index_val)
        context.nrt.incref(builder, signature.args[2], name_val)
        return series_val

    if is_heterogeneous_tuple_type(data):
        ret_typ = HeterogeneousSeriesType(data, index, name)
    else:
        dtype = data.dtype
        # XXX pd.DataFrame() calls init_series for even Series since it's untyped
        data = if_series_to_array_type(data)
        ret_typ = SeriesType(dtype, data, index, name)

    sig = signature(ret_typ, data, index, name)
    return sig, codegen


def init_series_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) >= 1 and not kws
    # TODO: add shape for index
    var = args[0]
    # avoid returning shape for tuple input (results in dimension mismatch error)
    data_type = self.typemap[var.name]
    if is_heterogeneous_tuple_type(data_type) or isinstance(data_type, types.BaseTuple):
        return None
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


from numba.parfors.array_analysis import ArrayAnalysis

ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_series_ext_init_series = (
    init_series_equiv
)


def get_series_payload(context, builder, series_type, value):
    meminfo = cgutils.create_struct_proxy(series_type)(context, builder, value).meminfo
    payload_type = SeriesPayloadType(series_type)
    payload = context.nrt.meminfo_data(builder, meminfo)
    ptrty = context.get_value_type(payload_type).as_pointer()
    payload = builder.bitcast(payload, ptrty)
    return context.make_helper(builder, payload_type, ref=payload)


@intrinsic
def _get_series_data(typingctx, series_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(context, builder, series_typ.data, series_payload.data)

    ret_typ = series_typ.data
    sig = signature(ret_typ, series_typ)
    return sig, codegen


@intrinsic
def _get_series_index(typingctx, series_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(
            context, builder, series_typ.index, series_payload.index
        )

    ret_typ = series_typ.index
    sig = signature(ret_typ, series_typ)
    return sig, codegen


@intrinsic
def _get_series_name(typingctx, series_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        # TODO: is borrowing None reference ok here?
        return impl_ret_borrowed(
            context, builder, signature.return_type, series_payload.name
        )

    sig = signature(series_typ.name_typ, series_typ)
    return sig, codegen


# this function should be used for getting S._data for alias analysis to work
# no_cpython_wrapper since Array(DatetimeDate) cannot be boxed
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_data(S):
    return lambda S: _get_series_data(S)


# TODO: use separate index type instead of just storing array
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_index(S):
    return lambda S: _get_series_index(S)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_name(S):
    return lambda S: _get_series_name(S)


# array analysis extension
def get_series_data_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 1 and not kws
    var = args[0]
    data_type = self.typemap[var.name].data
    # avoid returning shape for tuple input (results in dimension mismatch error)
    if is_heterogeneous_tuple_type(data_type) or isinstance(data_type, types.BaseTuple):
        return None
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


from numba.parfors.array_analysis import ArrayAnalysis

ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_series_ext_get_series_data = (
    get_series_data_equiv
)


def alias_ext_init_series(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)
    if len(args) > 1:  # has index
        numba.core.ir_utils._add_alias(lhs_name, args[1].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("init_series", "bodo.hiframes.pd_series_ext")
] = alias_ext_init_series


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("get_series_data", "bodo.hiframes.pd_series_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("get_series_index", "bodo.hiframes.pd_series_ext")
] = alias_ext_dummy_func


def series_to_array_type(typ):
    return typ.data
    # return _get_series_array_type(typ.dtype)


def is_series_type(typ):
    return isinstance(typ, SeriesType)


def if_series_to_array_type(typ):
    if isinstance(typ, SeriesType):
        return series_to_array_type(typ)

    return typ


@lower_cast(SeriesType, SeriesType)
def cast_series(context, builder, fromty, toty, val):
    # convert RangeIndex to NumericIndex if everything else is same
    if (
        fromty.copy(index=toty.index) == toty
        and isinstance(fromty.index, bodo.hiframes.pd_index_ext.RangeIndexType)
        and isinstance(toty.index, bodo.hiframes.pd_index_ext.NumericIndexType)
    ):

        series_payload = get_series_payload(context, builder, fromty, val)
        new_index = context.cast(
            builder, series_payload.index, fromty.index, toty.index
        )

        # increase refcount of stored values
        context.nrt.incref(builder, fromty.data, series_payload.data)
        context.nrt.incref(builder, fromty.name_typ, series_payload.name)

        return construct_series(
            context, builder, toty, series_payload.data, new_index, series_payload.name
        )

    return val


# --------------------------------------------------------------------------- #
# --- typing similar to arrays adopted from arraydecl.py, npydecl.py -------- #


@infer_getattr
class SeriesAttribute(AttributeTemplate):
    key = SeriesType

    def _resolve_map_func(self, ary, func, pysig, fname, f_args=None, kws=None):
        """Find type signature of Series.map/apply method.
        ary: Series type (TODO: rename)
        func: user-defined function
        pysig: python signature of the map/apply method
        fname: method name ("map" or "apply")
        f_args: arguments to UDF (only "apply" supports it)
        kws: kwargs to UDF (only "apply" supports it)
        """

        dtype = ary.dtype
        # TODO(ehsan): use getitem resolve similar to df.apply?
        # getitem returns Timestamp for dt_index and series(dt64)
        if dtype == types.NPDatetime("ns"):
            dtype = pd_timestamp_type
        # getitem returns Timedelta for td_index and series(td64)
        # TODO(ehsan): simpler to use timedelta64ns instead of types.NPTimedelta("ns")
        if dtype == types.NPTimedelta("ns"):
            dtype = pd_timedelta_type

        in_types = (dtype,)
        if f_args is not None:
            in_types += tuple(f_args.types)
        if kws is None:
            kws = {}
        return_nullable = False

        # Series.map() supports dictionary input
        if fname == "map" and isinstance(func, types.DictType):
            # TODO(ehsan): make sure dict key is comparable with input data type
            f_return_type = func.value_type
            return_nullable = True
        else:
            try:
                f_return_type = get_const_func_output_type(
                    func, in_types, kws, self.context
                )
            except Exception as e:
                raise BodoError(get_udf_error_msg(f"Series.{fname}()", e))

        if (
            isinstance(f_return_type, (SeriesType, HeterogeneousSeriesType))
            and f_return_type.const_info is None
        ):
            raise BodoError(
                "Invalid Series output in UDF (Series with constant length and constant Index value expected)"
            )

        # output is dataframe if UDF returns a Series
        if isinstance(f_return_type, HeterogeneousSeriesType):
            # NOTE: get_const_func_output_type() adds const_info attribute for Series
            # output
            _, index_vals = f_return_type.const_info
            arrs = tuple(_get_series_array_type(t) for t in f_return_type.data.types)
            ret_type = bodo.DataFrameType(arrs, ary.index, index_vals)
        elif isinstance(f_return_type, SeriesType):
            n_cols, index_vals = f_return_type.const_info
            arrs = tuple(
                _get_series_array_type(f_return_type.dtype) for _ in range(n_cols)
            )
            ret_type = bodo.DataFrameType(arrs, ary.index, index_vals)
        else:
            data_arr = get_udf_out_arr_type(f_return_type, return_nullable)
            ret_type = SeriesType(data_arr.dtype, data_arr, ary.index, ary.name_typ)

        return signature(ret_type, (func,)).replace(pysig=pysig)

    @bound_function("series.map", no_unliteral=True)
    def resolve_map(self, ary, args, kws):
        kws = dict(kws)
        func = args[0] if len(args) > 0 else kws["arg"]
        kws.pop("arg", None)
        na_action = args[1] if len(args) > 1 else kws.pop("na_action", types.none)

        unsupported_args = dict(
            na_action=na_action,
        )
        map_defaults = dict(
            na_action=None,
        )
        check_unsupported_args("Series.map", unsupported_args, map_defaults)

        def map_stub(arg, na_action=None):  # pragma: no cover
            pass

        pysig = numba.core.utils.pysignature(map_stub)
        return self._resolve_map_func(ary, func, pysig, "map")

    @bound_function("series.apply", no_unliteral=True)
    def resolve_apply(self, ary, args, kws):
        kws = dict(kws)
        func = args[0] if len(args) > 0 else kws["func"]
        kws.pop("func", None)
        convert_dtype = (
            args[1] if len(args) > 1 else kws.pop("convert_dtype", types.literal(True))
        )
        f_args = args[2] if len(args) > 2 else kws.pop("args", None)

        unsupported_args = dict(
            convert_dtype=convert_dtype,
        )
        apply_defaults = dict(
            convert_dtype=True,
        )
        check_unsupported_args("Series.apply", unsupported_args, apply_defaults)

        # add dummy default value for UDF kws to avoid errors
        kw_names = ", ".join("{} = ''".format(a) for a in kws.keys())
        func_text = f"def apply_stub(func, convert_dtype=True, args=(), {kw_names}):\n"
        func_text += "    pass\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        apply_stub = loc_vars["apply_stub"]

        pysig = numba.core.utils.pysignature(apply_stub)

        # TODO: handle apply differences: extra args, np ufuncs etc.
        return self._resolve_map_func(ary, func, pysig, "apply", f_args, kws)

    def _resolve_combine_func(self, ary, args, kws):
        # handle kwargs
        kwargs = dict(kws)
        other = args[0] if len(args) > 0 else types.unliteral(kwargs["other"])
        func = args[1] if len(args) > 1 else kwargs["func"]
        fill_value = (
            args[2]
            if len(args) > 2
            else types.unliteral(kwargs.get("fill_value", types.none))
        )

        def combine_stub(other, func, fill_value=None):  # pragma: no cover
            pass

        pysig = numba.core.utils.pysignature(combine_stub)

        # get return type
        dtype1 = ary.dtype
        # getitem returns Timestamp for dt_index and series(dt64)
        if dtype1 == types.NPDatetime("ns"):
            dtype1 = pd_timestamp_type
        dtype2 = other.dtype
        if dtype2 == types.NPDatetime("ns"):
            dtype2 = pd_timestamp_type

        f_return_type = get_const_func_output_type(
            func, (dtype1, dtype2), {}, self.context
        )

        # TODO: output name is always None in Pandas?
        sig = signature(
            SeriesType(f_return_type, index=ary.index, name_typ=types.none),
            (other, func, fill_value),
        )
        return sig.replace(pysig=pysig)

    @bound_function("series.combine", no_unliteral=True)
    def resolve_combine(self, ary, args, kws):
        return self._resolve_combine_func(ary, args, kws)

    def generic_resolve(self, S, attr):
        """Handle getattr on row Series values pass to df.apply() UDFs."""
        from bodo.hiframes.pd_index_ext import HeterogeneousIndexType

        if isinstance(S.index, HeterogeneousIndexType) and is_overload_constant_tuple(
            S.index.data
        ):
            indices = get_overload_const_tuple(S.index.data)
            if attr in indices:
                arr_ind = indices.index(attr)
                return S.data[arr_ind]


# pd.Series supports all operators except << and >>
series_binary_ops = tuple(
    op
    for op in numba.core.typing.npydecl.NumpyRulesArrayOperator._op_map.keys()
    if op not in (operator.lshift, operator.rshift)
)


# TODO: support itruediv, Numpy doesn't support it, and output can have
# a different type (output of integer division is float)
series_inplace_binary_ops = tuple(
    op
    for op in numba.core.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys()
    if op not in (operator.ilshift, operator.irshift, operator.itruediv)
)

inplace_binop_to_imm = {
    operator.iadd: operator.add,
    operator.isub: operator.sub,
    operator.imul: operator.mul,
    operator.ifloordiv: operator.floordiv,
    operator.imod: operator.mod,
    operator.ipow: operator.pow,
    operator.iand: operator.and_,
    operator.ior: operator.or_,
    operator.ixor: operator.xor,
}


series_unary_ops = (operator.neg, operator.invert, operator.pos)


str2str_methods = (
    "capitalize",
    "lower",
    "lstrip",
    "rstrip",
    "strip",
    "swapcase",
    "title",
    "upper",
)


str2bool_methods = (
    "isalnum",
    "isalpha",
    "isdigit",
    "isspace",
    "islower",
    "isupper",
    "istitle",
    "isnumeric",
    "isdecimal",
)


@overload(pd.Series, no_unliteral=True)
def pd_series_overload(
    data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
):
    # TODO: support isinstance in branch pruning pass
    # cases: dict, np.ndarray, Series, Index, arraylike (list, ...)

    # fastpath not supported
    if not is_overload_false(fastpath):
        raise BodoError("pd.Series(): 'fastpath' argument not supported.")

    # heterogeneous tuple input case
    if is_heterogeneous_tuple_type(data) and is_overload_none(dtype):

        def impl_heter(
            data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
        ):  # pragma: no cover
            index_t = bodo.utils.conversion.extract_index_if_none(data, index)
            data_t = bodo.utils.conversion.to_tuple(data)

            return bodo.hiframes.pd_series_ext.init_series(
                data_t, bodo.utils.conversion.convert_to_index(index_t), name
            )

        return impl_heter

    # support for series with no data
    if is_overload_none(data):

        def impl(
            data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
        ):  # pragma: no cover
            name_t = bodo.utils.conversion.extract_name_if_none(data, name)
            index_t = bodo.utils.conversion.extract_index_if_none(data, index)

            numba.parfors.parfor.init_prange()
            n = len(index_t)
            data_t = np.empty(n, np.float64)
            for i in numba.parfors.parfor.internal_prange(n):
                data_t[i] = np.nan

            return bodo.hiframes.pd_series_ext.init_series(
                data_t, bodo.utils.conversion.convert_to_index(index_t), name_t
            )

        return impl

    def impl(
        data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
    ):  # pragma: no cover
        # extract name if data is has name (Series/Index) and name is None
        name_t = bodo.utils.conversion.extract_name_if_none(data, name)
        index_t = bodo.utils.conversion.extract_index_if_none(data, index)
        data_t1 = bodo.utils.conversion.coerce_to_array(
            data, True, scalar_to_arr_len=len(index_t)
        )

        # TODO: support sanitize_array() of Pandas
        # TODO: add branch pruning to inline_closure_call
        # if dtype is not None:
        #     data_t2 = data_t1.astype(dtype)
        # else:
        #     data_t2 = data_t1

        # TODO: copy if index to avoid aliasing issues
        # data_t2 = data_t1
        data_t2 = bodo.utils.conversion.fix_arr_dtype(data_t1, dtype, None, False)

        # TODO: enable when branch pruning works for this
        # if copy:
        #     data_t2 = data_t1.copy()

        return bodo.hiframes.pd_series_ext.init_series(
            data_t2, bodo.utils.conversion.convert_to_index(index_t), name_t
        )

    return impl


@overload_method(SeriesType, "to_csv", no_unliteral=True)
def to_csv_overload(
    series,
    path_or_buf=None,
    sep=",",
    na_rep="",
    float_format=None,
    columns=None,
    header=True,
    index=True,
    index_label=None,
    mode="w",
    encoding=None,
    compression="infer",
    quoting=None,
    quotechar='"',
    line_terminator=None,
    chunksize=None,
    date_format=None,
    doublequote=True,
    escapechar=None,
    decimal=".",
    errors="strict",
    _is_parallel=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """Inspired by to_csv_overload in pd_dataframe_ext.py"""
    if not (
        is_overload_none(path_or_buf)
        or is_overload_constant_str(path_or_buf)
        or path_or_buf == string_type
    ):
        raise BodoError(
            "Series.to_csv(): 'path_or_buf' argument should be None or string"
        )

    if is_overload_none(path_or_buf):
        # String output case
        def _impl(
            series,
            path_or_buf=None,
            sep=",",
            na_rep="",
            float_format=None,
            columns=None,
            header=True,
            index=True,
            index_label=None,
            mode="w",
            encoding=None,
            compression="infer",
            quoting=None,
            quotechar='"',
            line_terminator=None,
            chunksize=None,
            date_format=None,
            doublequote=True,
            escapechar=None,
            decimal=".",
            errors="strict",
            _is_parallel=False,
        ):  # pragma: no cover
            with numba.objmode(D="unicode_type"):
                D = series.to_csv(
                    None,
                    sep,
                    na_rep,
                    float_format,
                    columns,
                    header,
                    index,
                    index_label,
                    mode,
                    encoding,
                    compression,
                    quoting,
                    quotechar,
                    line_terminator,
                    chunksize,
                    date_format,
                    doublequote,
                    escapechar,
                    decimal,
                    errors,
                )
            return D

        return _impl

    def _impl(
        series,
        path_or_buf=None,
        sep=",",
        na_rep="",
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        mode="w",
        encoding=None,
        compression="infer",
        quoting=None,
        quotechar='"',
        line_terminator=None,
        chunksize=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal=".",
        errors="strict",
        _is_parallel=False,
    ):  # pragma: no cover
        # passing None for the first argument returns a string
        # containing contents to write to csv
        if _is_parallel:
            header &= (bodo.libs.distributed_api.get_rank() == 0) | _csv_output_is_dir(
                unicode_to_utf8(path_or_buf)
            )
        with numba.objmode(D="unicode_type"):
            D = series.to_csv(
                None,
                sep,
                na_rep,
                float_format,
                columns,
                header,
                index,
                index_label,
                mode,
                encoding,
                compression,
                quoting,
                quotechar,
                line_terminator,
                chunksize,
                date_format,
                doublequote,
                escapechar,
                decimal,
                errors,
            )

        bodo.io.fs_io.csv_write(path_or_buf, D, _is_parallel)

    return _impl


# Raise Bodo Error for unsupported attributes and methods of Series
series_unsupported_attrs = (
    "array",  # TODO: support
    "at",
    "attrs",
    "axes",
    "nbytes",
    "is_unique",
    "sparse",
)


series_unsupported_methods = (
    "memory_usage",
    # Conversion
    "convert_dtypes",
    "infer_objects",
    "bool",
    "to_period",
    "to_timestamp",
    "__array__",
    # Indexing, iteration
    "get",
    "at",
    "__iter__",
    "items",
    "iteritems",
    "keys",
    "pop",
    "item",
    "xs",
    # Binary operator functions
    "combine_first",
    # Function application, groupby & window
    "agg",
    "aggregate",
    "transform",
    "expanding",
    "ewm",
    "pipe",
    # Computations / descriptive stats
    "clip",
    "factorize",
    "mode",
    "rank",
    # Reindexing / selection / label manipulation
    "align",
    "drop",
    "droplevel",
    "duplicated",
    "first",
    "last",
    "reindex",
    "reindex_like",
    "rename_axis",
    "sample",
    "set_axis",
    "truncate",
    "add_prefix",
    "add_suffix",
    "filter",
    # Missing data handling
    "backfill",
    "bfill",
    "ffill",
    "interpolate",
    "pad",
    # Reshaping, sorting
    "argmin",
    "argmax",
    "reorder_levels",
    "sort_index",
    "swaplevel",
    "unstack",
    "searchsorted",
    "ravel",
    "squeeze",
    "view",
    # Combining / joining / merging
    "compare",
    "update",
    # Time series-related
    "asfreq",
    "asof",
    "first_valid_index",
    "last_valid_index",
    "resample",
    "tz_convert",
    "tz_localize",
    "at_time",
    "between_time",
    "tshift",
    "slice_shift",
    # Plotting
    "plot",
    "hist",
    # Serialization / IO / conversion
    "to_pickle",
    "to_excel",
    "to_frame",
    "to_xarray",
    "to_hdf",
    "to_string",
    "to_clipboard",
    "to_latex",
    "to_markdown",
)


def _install_series_unsupported():
    """install an overload that raises BodoError for unsupported attributes and methods
    of Series
    """

    for attr_name in series_unsupported_attrs:
        full_name = "Series." + attr_name
        overload_attribute(SeriesType, attr_name)(
            create_unsupported_overload(full_name)
        )

    for fname in series_unsupported_methods:
        full_name = "Series." + fname
        overload_method(SeriesType, fname, no_unliteral=True)(
            create_unsupported_overload(full_name)
        )


_install_series_unsupported()
