# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implement pd.Series typing and data model handling.
"""
import operator
import pandas as pd
import numpy as np
import numba
from numba.core import types, cgutils
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
    overload_method,
    overload_attribute,
)
from numba.core.typing.templates import (
    infer_global,
    AbstractTemplate,
    signature,
    AttributeTemplate,
    bound_function,
)
from numba.core.typing.arraydecl import (
    get_array_index_type,
    _expand_integer,
    ArrayAttribute,
    SetItemBuffer,
)
from numba.core.typing.npydecl import (
    Numpy_rules_ufunc,
    NumpyRulesArrayOperator,
    NumpyRulesInplaceArrayOperator,
)
from numba.core.imputils import impl_ret_borrowed
from llvmlite import ir as lir

import bodo
from bodo.libs.str_ext import string_type
from bodo.libs.list_str_arr_ext import list_string_array_type
from bodo.libs.list_item_arr_ext import ListItemArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.int_arr_ext import IntegerArrayType, IntDtype
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.hiframes.pd_timestamp_ext import pandas_timestamp_type
from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.hiframes.pd_categorical_ext import PDCategoricalDtype, CategoricalArray
from bodo.hiframes.rolling import supported_rolling_funcs
from bodo.utils.typing import (
    is_overload_false,
    is_overload_none,
    create_unsupported_overload,
    BodoError,
)
from bodo.utils.transform import get_const_func_output_type


class SeriesType(types.IterableType, types.ArrayCompatible):
    """Temporary type class for Series objects.
    """

    def __init__(self, dtype, data=None, index=None, name_typ=None):
        from bodo.hiframes.pd_index_ext import RangeIndexType

        # keeping data array in type since operators can make changes such
        # as making array unaligned etc.
        # data is underlying array type and can have different dtype
        data = _get_series_array_type(dtype) if data is None else data
        # store regular dtype instead of IntDtype to avoid errors
        dtype = dtype.dtype if isinstance(dtype, IntDtype) else dtype
        # convert Record to tuple (for tuple output of map)
        # TODO: handle actual Record objects in Series?
        dtype = (
            types.Tuple(list(dict(dtype.members).values()))
            if isinstance(dtype, types.Record)
            else dtype
        )
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
        return types.iterators.ArrayIterator(self.data)


def _get_series_array_type(dtype):
    """get underlying default array type of series based on its dtype
    """
    # list(list(str))
    if dtype == types.List(string_type):
        # default data layout is list but split view is used if possible
        return list_string_array_type
    # string array
    elif dtype == string_type:
        return string_array_type
    elif isinstance(dtype, types.List):
        return ListItemArrayType(dtype.dtype)

    # categorical
    if isinstance(dtype, PDCategoricalDtype):
        return CategoricalArray(dtype)

    if isinstance(dtype, IntDtype):
        return IntegerArrayType(dtype.dtype)

    if dtype == types.bool_:
        return boolean_array

    # use recarray data layout for series of tuples
    if isinstance(dtype, types.BaseTuple):
        if any(not isinstance(t, types.Number) for t in dtype.types):
            # TODO: support more types. what types can be in recarrays?
            raise ValueError(
                "series tuple dtype {} includes non-numerics".format(dtype)
            )
        np_dtype = np.dtype(",".join(str(t) for t in dtype.types), align=True)
        dtype = numba.np.numpy_support.from_dtype(np_dtype)

    if dtype == datetime_date_type:
        return bodo.hiframes.datetime_date_ext.datetime_date_array_type

    if isinstance(dtype, Decimal128Type):
        return DecimalArrayType(dtype.precision, dtype.scale)

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
    ptrty = context.get_data_type(payload_type).as_pointer()
    payload_ptr = builder.bitcast(base_ptr, ptrty)
    payload = context.make_data_helper(builder, payload_type, ref=payload_ptr)

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
    payload_ll_type = context.get_data_type(payload_type)
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
    from bodo.hiframes.pd_multi_index_ext import MultiIndexType
    from bodo.hiframes.pd_index_ext import is_pd_index_type

    assert is_pd_index_type(index) or isinstance(index, MultiIndexType)
    name = types.none if name is None else name
    name = types.unliteral(name)

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
    if equiv_set.has_shape(var):
        return var, []
    return None


from numba.parfors.array_analysis import ArrayAnalysis

ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_series_ext_init_series = (
    init_series_equiv
)


def get_series_payload(context, builder, series_type, value):
    meminfo = cgutils.create_struct_proxy(series_type)(context, builder, value).meminfo
    payload_type = SeriesPayloadType(series_type)
    payload = context.nrt.meminfo_data(builder, meminfo)
    ptrty = context.get_data_type(payload_type).as_pointer()
    payload = builder.bitcast(payload, ptrty)
    return context.make_data_helper(builder, payload_type, ref=payload)


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
    if equiv_set.has_shape(var):
        return var, []
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


@numba.njit
def convert_index_to_int64(S):  # pragma: no cover
    # convert Series with none index to numeric index
    data = bodo.hiframes.pd_series_ext.get_series_data(S)
    index_arr = np.arange(len(data))
    name = bodo.hiframes.pd_series_ext.get_series_name(S)
    return bodo.hiframes.pd_series_ext.init_series(
        data, bodo.utils.conversion.convert_to_index(index_arr), name
    )


# cast Series(int8) to Series(cat) for init_series() in test_csv_cat1
# TODO: separate array type for Categorical data
@lower_cast(SeriesType, SeriesType)
def cast_series(context, builder, fromty, toty, val):
    # convert None index to Integer index if everything else is same
    if (
        fromty.copy(index=toty.index) == toty
        and fromty.index == types.none
        and toty.index
        == bodo.hiframes.pd_index_ext.NumericIndexType(types.int64, types.none)
    ):
        cn_typ = context.typing_context.resolve_value_type(convert_index_to_int64)
        cn_sig = cn_typ.get_call_type(context.typing_context, (fromty,), {})
        cn_impl = context.get_function(cn_typ, cn_sig)
        return cn_impl(builder, (val,))

    return val


# --------------------------------------------------------------------------- #
# --- typing similar to arrays adopted from arraydecl.py, npydecl.py -------- #


@infer_getattr
class SeriesAttribute(AttributeTemplate):
    key = SeriesType

    @bound_function("series.rolling", no_unliteral=True)
    def resolve_rolling(self, ary, args, kws):
        def rolling_stub(
            window,
            min_periods=None,
            center=False,
            win_type=None,
            on=None,
            axis=0,
            closed=None,
        ):  # pragma: no cover
            pass

        pysig = numba.core.utils.pysignature(rolling_stub)
        return signature(SeriesRollingType(ary), *args).replace(pysig=pysig)

    def _resolve_map_func(self, ary, func, pysig, fname, f_args=None):
        """Find type signature of Series.map/apply method.
        ary: Series type (TODO: rename)
        func: user-defined function
        pysig: python signature of the map/apply method
        fname: method name ("map" or "apply")
        f_args: arguments to UDF (only "apply" supports it)
        """

        dtype = ary.dtype
        # getitem returns Timestamp for dt_index and series(dt64)
        if dtype == types.NPDatetime("ns"):
            dtype = pandas_timestamp_type

        in_types = (dtype,)
        if f_args is not None:
            in_types += tuple(f_args.types)

        try:
            f_return_type = get_const_func_output_type(func, in_types, self.context)
        except:
            raise BodoError(f"Series.{fname}(): user-defined function not supported")

        # unbox Timestamp to dt64 in Series (TODO: timedelta64)
        if f_return_type == pandas_timestamp_type:
            f_return_type = types.NPDatetime("ns")

        data_arr = _get_series_array_type(f_return_type)
        # Series.map codegen returns np bool array instead of boolean_array currently
        # TODO: return nullable boolean_array
        if f_return_type == types.bool_:
            data_arr = types.Array(types.bool_, 1, "C")
        return signature(
            SeriesType(f_return_type, data_arr, ary.index, ary.name_typ), (func,)
        ).replace(pysig=pysig)

    @bound_function("series.map", no_unliteral=True)
    def resolve_map(self, ary, args, kws):
        kwargs = dict(kws)
        func = args[0] if len(args) > 0 else kwargs["arg"]

        def map_stub(arg, na_action=None):  # pragma: no cover
            pass

        pysig = numba.core.utils.pysignature(map_stub)
        return self._resolve_map_func(ary, func, pysig, "map")

    @bound_function("series.apply", no_unliteral=True)
    def resolve_apply(self, ary, args, kws):
        kwargs = dict(kws)
        func = args[0] if len(args) > 0 else kwargs["func"]
        f_args = args[2] if len(args) > 2 else kwargs.pop("args", None)

        def apply_stub(func, convert_dtype=True, args=()):  # pragma: no cover
            pass

        pysig = numba.core.utils.pysignature(apply_stub)
        # TODO: handle apply differences: extra args, np ufuncs etc.
        return self._resolve_map_func(ary, func, pysig, "apply", f_args)

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
            dtype1 = pandas_timestamp_type
        dtype2 = other.dtype
        if dtype2 == types.NPDatetime("ns"):
            dtype2 = pandas_timestamp_type

        f_return_type = get_const_func_output_type(func, (dtype1, dtype2), self.context)

        # TODO: output name is always None in Pandas?
        sig = signature(
            SeriesType(f_return_type, index=ary.index, name_typ=types.none),
            (other, func, fill_value),
        )
        return sig.replace(pysig=pysig)

    @bound_function("series.combine", no_unliteral=True)
    def resolve_combine(self, ary, args, kws):
        return self._resolve_combine_func(ary, args, kws)

    # TODO: use overload when Series.aggregate is supported
    @bound_function("series.value_counts", no_unliteral=True)
    def resolve_value_counts(self, ary, args, kws):
        # output is int series with original data as index
        index_typ = bodo.hiframes.pd_index_ext.array_typ_to_index(ary.data)
        out = SeriesType(
            types.int64, types.Array(types.int64, 1, "C"), index_typ, ary.name_typ
        )
        return signature(out, *args)


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


class SeriesRollingType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = "SeriesRollingType({})".format(stype)
        super(SeriesRollingType, self).__init__(name)


@infer_getattr
class SeriesRollingAttribute(AttributeTemplate):
    key = SeriesRollingType

    @bound_function("rolling.apply", no_unliteral=True)
    def resolve_apply(self, ary, args, kws):
        # result is always float64 (see Pandas window.pyx:roll_generic)
        return signature(
            SeriesType(
                types.float64, index=ary.stype.index, name_typ=ary.stype.name_typ
            ),
            *args,
        )

    @bound_function("rolling.cov", no_unliteral=True)
    def resolve_cov(self, ary, args, kws):
        return signature(
            SeriesType(
                types.float64, index=ary.stype.index, name_typ=ary.stype.name_typ
            ),
            *args,
        )

    @bound_function("rolling.corr", no_unliteral=True)
    def resolve_corr(self, ary, args, kws):
        return signature(
            SeriesType(
                types.float64, index=ary.stype.index, name_typ=ary.stype.name_typ
            ),
            *args,
        )


# similar to install_array_method in arraydecl.py
def install_rolling_method(name):
    def rolling_attribute_attachment(self, ary):
        def rolling_generic(self, args, kws):
            # output is always float64
            return signature(
                SeriesType(
                    types.float64, index=ary.stype.index, name_typ=ary.stype.name_typ
                ),
                *args,
            )

        my_attr = {"key": "rolling." + name, "generic": rolling_generic}
        temp_class = type("Rolling_" + name, (AbstractTemplate,), my_attr)
        return types.BoundFunction(temp_class, ary)

    setattr(SeriesRollingAttribute, "resolve_" + name, rolling_attribute_attachment)


for fname in supported_rolling_funcs:
    install_rolling_method(fname)


@infer
@infer_global(operator.eq)
@infer_global(operator.ne)
@infer_global(operator.ge)
@infer_global(operator.gt)
@infer_global(operator.le)
@infer_global(operator.lt)
class SeriesCompEqual(AbstractTemplate):
    key = "=="

    def generic(self, args, kws):
        from bodo.libs.str_arr_ext import is_str_arr_typ

        assert not kws
        [va, vb] = args
        # if one of the inputs is string array
        if is_str_series_typ(va) or is_str_series_typ(vb):
            # inputs should be either string array or string
            assert is_str_arr_typ(va) or va == string_type
            assert is_str_arr_typ(vb) or vb == string_type
            return signature(SeriesType(types.boolean, boolean_array), va, vb)

        if (is_dt64_series_typ(va) and vb in (string_type, types.NPDatetime("ns"))) or (
            is_dt64_series_typ(vb) and va in (string_type, types.NPDatetime("ns"))
        ):
            if is_dt64_series_typ(va):
                index_typ = va.index
            else:
                index_typ = vb.index
            return signature(
                SeriesType(types.boolean, boolean_array, index_typ), va, vb
            )

        if is_dt64_series_typ(va) and is_dt64_series_typ(vb):
            return signature(SeriesType(types.boolean, boolean_array, va.index), va, vb)


@infer
class CmpOpNEqSeries(SeriesCompEqual):
    key = "!="


@infer
class CmpOpGESeries(SeriesCompEqual):
    key = ">="


@infer
class CmpOpGTSeries(SeriesCompEqual):
    key = ">"


@infer
class CmpOpLESeries(SeriesCompEqual):
    key = "<="


@infer
class CmpOpLTSeries(SeriesCompEqual):
    key = "<"


@overload(pd.Series, no_unliteral=True)
def pd_series_overload(
    data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
):

    # TODO: support isinstance in branch pruning pass
    # cases: dict, np.ndarray, Series, Index, arraylike (list, ...)

    # TODO: None or empty data
    if is_overload_none(data):
        raise ValueError("pd.Series(): 'data' argument required.")

    # fastpath not supported
    if not is_overload_false(fastpath):
        raise ValueError("pd.Series(): 'fastpath' argument not supported.")

    def impl(
        data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
    ):  # pragma: no cover
        # extract name if data is has name (Series/Index) and name is None
        name_t = bodo.utils.conversion.extract_name_if_none(data, name)
        index_t = bodo.utils.conversion.extract_index_if_none(data, index)
        data_t1 = bodo.utils.conversion.coerce_to_array(data, True)

        # TODO: support sanitize_array() of Pandas
        # TODO: add branch pruning to inline_closure_call
        # if dtype is not None:
        #     data_t2 = data_t1.astype(dtype)
        # else:
        #     data_t2 = data_t1

        # TODO: copy if index to avoid aliasing issues
        # data_t2 = data_t1
        data_t2 = bodo.utils.conversion.fix_arr_dtype(data_t1, dtype)

        # TODO: enable when branch pruning works for this
        # if copy:
        #     data_t2 = data_t1.copy()

        return bodo.hiframes.pd_series_ext.init_series(
            data_t2, bodo.utils.conversion.convert_to_index(index_t), name_t
        )

    return impl


def get_series_data_tup(series_tup):  # pragma: no cover
    return tuple(get_series_data(s) for s in series_tup)


@overload(get_series_data_tup, no_unliteral=True)
def overload_get_series_data_tup(series_tup):
    n_series = len(series_tup.types)
    func_text = "def f(series_tup):\n"
    res = ",".join(
        "bodo.hiframes.pd_series_ext.get_series_data(series_tup[{}])".format(i)
        for i in range(n_series)
    )
    func_text += "  return ({}{})\n".format(res, "," if n_series == 1 else "")
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    impl = loc_vars["f"]
    return impl


# Raise Bodo Error for unsupported attributes and methods of Series
series_unsupported_attrs = (
    "array",  # TODO: support
    "nbytes",
    "is_unique",
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
    "cat",
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
    "groupby",
    "expanding",
    "ewm",
    "pipe",
    # Computations / descriptive stats
    "autocorr",
    "between",
    "clip",
    "diff",
    "factorize",
    "mode",
    "rank",
    # Reindexing / selection / label manipulation
    "align",
    "drop",
    "droplevel",
    "drop_duplicates",
    "duplicated",
    "first",
    "last",
    "reindex",
    "reindex_like",
    "rename_axis",
    "sample",
    "set_axis",
    "truncate",
    "where",
    "mask",
    "add_prefix",
    "add_suffix",
    "filter",
    # Missing data handling
    "interpolate",
    # Reshaping, sorting
    "argmin",
    "argmax",
    "reorder_levels",
    "sort_index",
    "swaplevel",
    "unstack",
    "explode",
    "searchsorted",
    "ravel",
    "repeat",
    "squeeze",
    "view",
    # Combining / joining / merging
    "replace",
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
    "to_dict",
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
