"""Implement pd.Series typing and data model handling."""

import operator

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed, lower_constant
from numba.core.typing.templates import bound_function, signature
from numba.extending import (
    infer_getattr,
    intrinsic,
    lower_builtin,
    lower_cast,
    models,
    overload,
    overload_method,
    register_model,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
import bodo.pandas as bd
from bodo.hiframes.datetime_date_ext import datetime_date_type
from bodo.hiframes.datetime_timedelta_ext import pd_timedelta_type
from bodo.hiframes.pd_timestamp_ext import (
    PandasTimestampType,
    pd_timestamp_tz_naive_type,
)
from bodo.io import csv_cpp
from bodo.ir.unsupported_method_template import (
    overload_unsupported_attribute,
    overload_unsupported_method,
)
from bodo.libs.float_arr_ext import FloatDtype
from bodo.libs.int_arr_ext import IntDtype
from bodo.libs.pd_datetime_arr_ext import PandasDatetimeTZDtype
from bodo.libs.str_ext import string_type, unicode_to_utf8
from bodo.utils.templates import OverloadedKeyAttributeTemplate
from bodo.utils.transform import get_const_func_output_type
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    dtype_to_array_type,
    get_overload_const_str,
    get_overload_const_tuple,
    get_udf_error_msg,
    get_udf_out_arr_type,
    is_heterogeneous_tuple_type,
    is_overload_constant_str,
    is_overload_constant_tuple,
    is_overload_false,
    is_overload_int,
    is_overload_none,
    raise_bodo_error,
    to_nullable_type,
)

_csv_output_is_dir = types.ExternalFunction(
    "csv_output_is_dir",
    types.int8(types.voidptr),
)
ll.add_symbol("csv_output_is_dir", csv_cpp.csv_output_is_dir)


class SeriesType(types.IterableType, types.ArrayCompatible):
    """Type class for Series objects"""

    ndim = 1

    def __init__(self, dtype, data=None, index=None, name_typ=None, dist=None):
        from bodo.hiframes.pd_index_ext import RangeIndexType
        from bodo.transforms.distributed_analysis import Distribution

        # keeping data array in type since operators can make changes such
        # as making array unaligned etc.
        # data is underlying array type and can have different dtype
        data = dtype_to_array_type(dtype) if data is None else data
        # store regular dtype instead of IntDtype to avoid errors
        dtype = dtype.dtype if isinstance(dtype, IntDtype) else dtype
        # store regular dtype instead of FloatDtype to avoid errors
        dtype = dtype.dtype if isinstance(dtype, FloatDtype) else dtype
        self.dtype = dtype
        self.data = data
        name_typ = types.none if name_typ is None else name_typ
        index = RangeIndexType(types.none) if index is None else index
        self.index = index  # index should be an Index type (not Array)
        self.name_typ = name_typ
        # see comment on 'dist' in DataFrameType
        dist = Distribution.OneD_Var if dist is None else dist
        self.dist = dist
        super().__init__(name=f"series({dtype}, {data}, {index}, {name_typ}, {dist})")

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self, dtype=None, index=None, dist=None):
        # XXX is copy necessary?
        if index is None:
            index = self.index
        if dist is None:
            dist = self.dist
        if dtype is None:
            dtype = self.dtype
            data = self.data
        else:
            data = dtype_to_array_type(dtype)
        return SeriesType(dtype, data, index, self.name_typ, dist)

    @property
    def key(self):
        # needed?
        return self.dtype, self.data, self.index, self.name_typ, self.dist

    def unify(self, typingctx, other):
        from bodo.transforms.distributed_analysis import Distribution

        if isinstance(other, SeriesType):
            # NOTE: checking equality since Index types may not have unify() implemented
            # TODO: add unify() to all Index types and remove this
            new_index = (
                self.index
                if self.index == other.index
                else self.index.unify(typingctx, other.index)
            )
            # use the most conservative distribution
            dist = Distribution(min(self.dist.value, other.dist.value))

            # If dtype matches or other.dtype is undefined (inferred)
            if other.dtype == self.dtype or not other.dtype.is_precise():
                return SeriesType(
                    self.dtype,
                    self.data
                    if self.data == other.data
                    else self.data.unify(typingctx, other.data),
                    new_index,
                    dist=dist,
                )

        # XXX: unify Series/Array as Array
        return super().unify(typingctx, other)

    def can_convert_to(self, typingctx, other):
        from numba.core.typeconv import Conversion

        if (
            isinstance(other, SeriesType)
            and self.dtype == other.dtype
            and self.data == other.data
            and self.index == other.index
            and self.name_typ == other.name_typ
            and self.dist != other.dist
        ):
            return Conversion.safe

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

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


class HeterogeneousSeriesType(types.Type):
    """
    Type class for Series objects with heterogeneous values (e.g. pd.Series([1, 'A']))
    """

    ndim = 1

    def __init__(self, data=None, index=None, name_typ=None):
        from bodo.hiframes.pd_index_ext import RangeIndexType
        from bodo.transforms.distributed_analysis import Distribution

        self.data = data
        name_typ = types.none if name_typ is None else name_typ
        index = RangeIndexType(types.none) if index is None else index
        # TODO(ehsan): add check for index type
        self.index = index  # index should be an Index type (not Array)
        self.name_typ = name_typ
        self.dist = Distribution.REP  # cannot be distributed
        super().__init__(name=f"heter_series({data}, {index}, {name_typ})")

    def copy(self, index=None, dist=None):
        # 'dist' argument is necessary since distributed analysis calls copy() for
        # potential distribution updates (with hasattr(typ, "dist") check)
        from bodo.transforms.distributed_analysis import Distribution

        assert dist == Distribution.REP, (
            "invalid distribution for HeterogeneousSeriesType"
        )

        if index is None:
            index = self.index.copy()
        return HeterogeneousSeriesType(self.data, index, self.name_typ)

    @property
    def key(self):
        return self.data, self.index, self.name_typ

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@lower_builtin("getiter", SeriesType)
def series_getiter(context, builder, sig, args):
    """support getting an iterator object for Series by calling 'getiter' on the
    underlying array.
    """
    series_payload = get_series_payload(context, builder, sig.args[0], args[0])
    impl = context.get_function("getiter", sig.return_type(sig.args[0].data))
    return impl(builder, (series_payload.data,))


@infer_getattr
class HeterSeriesAttribute(OverloadedKeyAttributeTemplate):
    key = HeterogeneousSeriesType

    def generic_resolve(self, S, attr):
        """Handle getattr on row Series values pass to df.apply() UDFs."""
        from bodo.hiframes.pd_index_ext import HeterogeneousIndexType

        # If an column name conflicts with a Series method/attribute we
        # shouldn't search the columns.
        if self._is_existing_attr(attr):
            return

        if isinstance(S.index, HeterogeneousIndexType) and is_overload_constant_tuple(
            S.index.data
        ):
            indices = get_overload_const_tuple(S.index.data)
            if attr in indices:
                arr_ind = indices.index(attr)
                return S.data[arr_ind]


def is_str_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == string_type


def is_dt64_series_typ(t):
    return isinstance(t, SeriesType) and (
        t.dtype == types.NPDatetime("ns") or isinstance(t.dtype, PandasDatetimeTZDtype)
    )


def is_timedelta64_series_typ(t):
    return isinstance(t, SeriesType) and (
        t.dtype == types.NPTimedelta("ns") or t.data == bodo.types.timedelta_array_type
    )


def is_datetime_date_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == datetime_date_type


# payload type inside meminfo so that mutation are seen by all references
class SeriesPayloadType(types.Type):
    def __init__(self, series_type):
        self.series_type = series_type
        super().__init__(name=f"SeriesPayloadType({series_type})")

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@register_model(SeriesPayloadType)
class SeriesPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.series_type.data),
            ("index", fe_type.series_type.index),
            ("name", fe_type.series_type.name_typ),
        ]
        super().__init__(dmm, fe_type, members)


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
        super().__init__(dmm, fe_type, members)


def define_series_dtor(context, builder, series_type, payload_type):
    """
    Define destructor for Series type if not already defined
    """
    mod = builder.module
    # Declare dtor
    fnty = lir.FunctionType(lir.VoidType(), [cgutils.voidptr_t])
    # TODO(ehsan): do we need to sanitize the name in any case?
    fn = cgutils.get_or_insert_function(mod, fnty, name=f".dtor.series.{series_type}")

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


@intrinsic(prefer_literal=True)
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
    """array analysis for init_series(), which inserts equivalence for input data array,
    input index and output Series.
    """
    from bodo.hiframes.pd_index_ext import HeterogeneousIndexType

    assert len(args) >= 2 and not kws
    data = args[0]
    index = args[1]

    # avoid returning shape for tuple input (results in dimension mismatch error)
    data_type = self.typemap[data.name]
    if is_heterogeneous_tuple_type(data_type) or isinstance(data_type, types.BaseTuple):
        return None

    # index and data have the same length (avoid tuple index)
    index_type = self.typemap[index.name]
    if (
        not isinstance(index_type, HeterogeneousIndexType)
        and equiv_set.has_shape(data)
        and equiv_set.has_shape(index)
    ):
        equiv_set.insert_equiv(data, index)

    if equiv_set.has_shape(data):
        return ArrayAnalysis.AnalyzeResult(shape=data, pre=[])
    return None


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
def get_series_data(typingctx, series_typ):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(context, builder, series_typ.data, series_payload.data)

    ret_typ = series_typ.data
    sig = signature(ret_typ, series_typ)
    return sig, codegen


@intrinsic
def get_series_index(typingctx, series_typ):
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
def get_series_name(typingctx, series_typ):
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


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_series_ext_get_series_data = (
    get_series_data_equiv
)


def get_series_index_equiv(self, scope, equiv_set, loc, args, kws):
    """array analysis equivalence extension for get_series_index()"""
    from bodo.hiframes.pd_index_ext import HeterogeneousIndexType

    assert len(args) == 1 and not kws
    var = args[0]
    index_type = self.typemap[var.name].index
    # avoid returning shape for tuple input (results in dimension mismatch error)
    if isinstance(index_type, HeterogeneousIndexType):
        return None
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_series_ext_get_series_index = (
    get_series_index_equiv
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


def is_series_type(typ):
    return isinstance(typ, SeriesType)


def if_series_to_array_type(typ):
    if isinstance(typ, SeriesType):
        return typ.data

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

    # trivial cast if only 'dist' is different (no need to change value)
    if (
        fromty.dtype == toty.dtype
        and fromty.data == toty.data
        and fromty.index == toty.index
        and fromty.name_typ == toty.name_typ
        and fromty.dist != toty.dist
    ):
        return val

    # TODO(ehsan): is this safe?
    return val


# --------------------------------------------------------------------------- #
# --- typing similar to arrays adopted from arraydecl.py, npydecl.py -------- #


@infer_getattr
class SeriesAttribute(OverloadedKeyAttributeTemplate):
    key = SeriesType

    @bound_function("series.head")
    def resolve_head(self, ary, args, kws):
        func_name = "Series.head"

        # Obtain a the pysig and folded args
        arg_names = ("n",)
        arg_defaults = {"n": 5}

        pysig, folded_args = bodo.utils.typing.fold_typing_args(
            func_name, args, kws, arg_names, arg_defaults
        )

        # Check typing on arguments
        n_arg = folded_args[0]
        if not is_overload_int(n_arg):
            raise BodoError(f"{func_name}(): 'n' must be an Integer")

        # Determine the return type
        # Return type is the same as the series
        ret = ary
        # Return the signature
        return ret(*folded_args).replace(pysig=pysig)

    def _resolve_map_func(
        self, ary, func, pysig, fname, f_args=None, kws=None, na_action=None
    ):
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
            dtype = pd_timestamp_tz_naive_type
        # getitem returns Timedelta for td_index and series(td64)
        # TODO(ehsan): simpler to use timedelta64ns instead of types.NPTimedelta("ns")
        if dtype == types.NPTimedelta("ns"):
            dtype = pd_timedelta_type
        if isinstance(dtype, PandasDatetimeTZDtype):
            dtype = PandasTimestampType(dtype.tz)

        in_types = (dtype,)
        if f_args is not None:
            in_types += tuple(f_args.types)
        if kws is None:
            kws = {}
        return_nullable = False

        # The output may contain NAs from input in this case
        if na_action == "ignore":
            return_nullable = True

        # Is the function a UDF or a builtin
        is_udf = True

        # Series.map() supports dictionary input
        if fname == "map" and isinstance(func, types.DictType):
            # TODO(ehsan): make sure dict key is comparable with input data type
            f_return_type = func.value_type
            return_nullable = True
        else:
            try:
                if types.unliteral(func) == types.unicode_type:
                    if not is_overload_constant_str(func):
                        raise BodoError(
                            "Series.apply(): string argument (for builtins) must be a compile time constant"
                        )
                    f_return_type = bodo.utils.transform.get_udf_str_return_type(
                        ary,
                        get_overload_const_str(func),
                        self.context,
                        "Series.apply",
                    )
                    is_udf = False
                elif bodo.utils.typing.is_numpy_ufunc(func):
                    f_return_type = func.get_call_type(
                        self.context,
                        # We don't support only args/kws other than the Series yet
                        (ary,),
                        {},
                    ).return_type
                    is_udf = False
                else:
                    f_return_type = get_const_func_output_type(
                        func,
                        in_types,
                        kws,
                        self.context,
                        numba.core.registry.cpu_target.target_context,
                    )
            except Exception as e:
                raise BodoError(get_udf_error_msg(f"Series.{fname}()", e))

        if is_udf:
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
                # Heterogenous Series should always return a Nullable Tuple in the output type,
                if isinstance(
                    f_return_type.data, bodo.libs.nullable_tuple_ext.NullableTupleType
                ):
                    scalar_types = f_return_type.data.tuple_typ.types
                elif isinstance(f_return_type.data, types.Tuple):
                    # TODO: Confirm if this path ever taken? It shouldn't be.
                    scalar_types = f_return_type.data.types
                # NOTE: nullable is determined at runtime, so by default always assume nullable type
                # TODO: Support for looking at constant values.
                arrs = tuple(
                    to_nullable_type(dtype_to_array_type(t)) for t in scalar_types
                )
                ret_type = bodo.types.DataFrameType(arrs, ary.index, index_vals)
            elif isinstance(f_return_type, SeriesType):
                n_cols, index_vals = f_return_type.const_info
                # Note: For homogenous Series we return a regular tuple, so
                # convert to nullable.
                # NOTE: nullable is determined at runtime, so by default always assume nullable type
                # TODO: Support for looking at constant values.
                arrs = tuple(
                    to_nullable_type(dtype_to_array_type(f_return_type.dtype))
                    for _ in range(n_cols)
                )
                ret_type = bodo.types.DataFrameType(arrs, ary.index, index_vals)
            else:
                data_arr = get_udf_out_arr_type(f_return_type, return_nullable)
                ret_type = SeriesType(data_arr.dtype, data_arr, ary.index, ary.name_typ)
        else:
            # If apply just calls a builtin function we just return the type of that
            # function.
            ret_type = f_return_type

        return signature(ret_type, (func,)).replace(pysig=pysig)

    @bound_function("series.map", no_unliteral=True)
    def resolve_map(self, ary, args, kws):
        kws = dict(kws)
        func = args[0] if len(args) > 0 else kws["arg"]
        kws.pop("arg", None)
        na_action = args[1] if len(args) > 1 else kws.pop("na_action", types.none)
        if not (
            is_overload_none(na_action)
            or (
                is_overload_constant_str(na_action)
                and (get_overload_const_str(na_action) == "ignore")
            )
        ):
            raise BodoError(
                "Series.map(): 'na_action' must be None or constant string 'ignore'"
            )
        na_action = (
            get_overload_const_str(na_action)
            if is_overload_constant_str(na_action)
            else None
        )
        # NOTE: pandas doesn't support args but we support it to make UDF engine simpler
        f_args = args[2] if len(args) > 2 else kws.pop("args", None)

        # add dummy default value for UDF kws to avoid errors
        kw_names = ", ".join(f"{a} = ''" for a in kws.keys())
        func_text = f"def map_stub(arg, na_action=None, args=(), {kw_names}):\n"
        func_text += "    pass\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        map_stub = loc_vars["map_stub"]

        pysig = numba.core.utils.pysignature(map_stub)

        return self._resolve_map_func(ary, func, pysig, "map", f_args, kws, na_action)

    @bound_function("series.apply", no_unliteral=True)
    def resolve_apply(self, ary, args, kws):
        kws = dict(kws)
        func = args[0] if len(args) > 0 else kws["func"]
        kws.pop("func", None)
        convert_dtype = (
            args[1] if len(args) > 1 else kws.pop("convert_dtype", types.literal(True))
        )
        f_args = args[2] if len(args) > 2 else kws.pop("args", None)

        unsupported_args = {"convert_dtype": convert_dtype}
        apply_defaults = {"convert_dtype": True}
        check_unsupported_args(
            "Series.apply",
            unsupported_args,
            apply_defaults,
            package_name="pandas",
            module_name="Series",
        )

        # add dummy default value for UDF kws to avoid errors
        kw_names = ", ".join(f"{a} = ''" for a in kws.keys())
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
        bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
            ary, "Series.combine()"
        )
        bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
            other, "Series.combine()"
        )
        if dtype1 == types.NPDatetime("ns"):
            dtype1 = pd_timestamp_tz_naive_type
        dtype2 = other.dtype
        if dtype2 == types.NPDatetime("ns"):
            dtype2 = pd_timestamp_tz_naive_type

        f_return_type = get_const_func_output_type(
            func,
            (dtype1, dtype2),
            {},
            self.context,
            numba.core.registry.cpu_target.target_context,
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

    @bound_function("series.pipe", no_unliteral=True)
    def resolve_pipe(self, ary, args, kws):
        return bodo.hiframes.pd_groupby_ext.resolve_obj_pipe(
            self, ary, args, kws, "Series"
        )

    def generic_resolve(self, S, attr):
        """Handle getattr on row Series values pass to df.apply() UDFs."""
        from bodo.hiframes.pd_index_ext import HeterogeneousIndexType

        # If an column name conflicts with a Series method/attribute we
        # shouldn't search the columns.
        if self._is_existing_attr(attr):
            return

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
@overload(bd.Series, no_unliteral=True)
def pd_series_overload(
    data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
):
    # TODO: support isinstance in branch pruning pass
    # cases: dict, np.ndarray, Series, Index, arraylike (list, ...)

    # fastpath not supported
    if not is_overload_false(fastpath):
        raise BodoError("pd.Series(): 'fastpath' argument not supported.")

    no_data = is_overload_none(data)
    no_index = is_overload_none(index)
    no_dtype = is_overload_none(dtype)
    if no_data and no_index and no_dtype:
        raise BodoError(
            "pd.Series() requires at least 1 of data, index, and dtype to not be none"
        )

    if is_series_type(data) and not no_index:
        raise BodoError(
            "pd.Series() does not support index value when input data is a Series"
        )

    # In the case that we have a dictionary input type, we need to wait for a transformation in typing pass
    # to convert it to a tuple type, so we can access the keys as constants
    if isinstance(data, types.DictType):
        raise_bodo_error(
            "pd.Series(): When initializing series with a dictionary, it is required that the dict has constant keys"
        )

    # heterogeneous tuple input case
    if is_heterogeneous_tuple_type(data) and is_overload_none(dtype):
        # Generate a null tuple so all Heterogenous tuple Series create a
        # nullable tuple
        null_tup = tuple(len(data) * [False])

        def impl_heter(
            data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
        ):  # pragma: no cover
            index_t = bodo.utils.conversion.extract_index_if_none(data, index)
            data_t = bodo.utils.conversion.to_tuple(data)
            data_val = bodo.libs.nullable_tuple_ext.build_nullable_tuple(
                data_t, null_tup
            )
            return bodo.hiframes.pd_series_ext.init_series(
                data_val, bodo.utils.conversion.convert_to_index(index_t), name
            )

        return impl_heter

    # support for series with no data
    if no_data:
        if no_dtype:

            def impl(
                data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
            ):  # pragma: no cover
                name_t = bodo.utils.conversion.extract_name_if_none(data, name)
                index_t = bodo.utils.conversion.extract_index_if_none(data, index)

                numba.parfors.parfor.init_prange()
                n = len(index_t)
                data_t = np.empty(n, np.float64)
                for i in numba.parfors.parfor.internal_prange(n):
                    bodo.libs.array_kernels.setna(data_t, i)

                return bodo.hiframes.pd_series_ext.init_series(
                    data_t, bodo.utils.conversion.convert_to_index(index_t), name_t
                )

            return impl

        # If a dtype is provided we need to convert the passed dtype into an array_type
        if bodo.utils.conversion._is_str_dtype(dtype):
            _arr_dtype = bodo.types.string_array_type
        else:
            nb_dtype = bodo.utils.typing.parse_dtype(dtype, "pandas.Series")
            if isinstance(nb_dtype, bodo.libs.int_arr_ext.IntDtype):
                _arr_dtype = bodo.types.IntegerArrayType(nb_dtype.dtype)
            elif isinstance(nb_dtype, bodo.libs.float_arr_ext.FloatDtype):
                _arr_dtype = bodo.types.FloatingArrayType(nb_dtype.dtype)
            elif nb_dtype == bodo.libs.bool_arr_ext.boolean_dtype:
                _arr_dtype = bodo.types.boolean_array_type
            elif nb_dtype == bodo.types.datetime64ns:  # pragma: no cover
                _arr_dtype = bodo.libs.pd_datetime_arr_ext.DatetimeArrayType(None)
            elif isinstance(nb_dtype, PandasDatetimeTZDtype):
                _arr_dtype = bodo.libs.pd_datetime_arr_ext.DatetimeArrayType(
                    nb_dtype.tz
                )
            elif (
                isinstance(nb_dtype, types.Number)
                or nb_dtype == bodo.types.timedelta64ns
            ):
                _arr_dtype = types.Array(nb_dtype, 1, "C")
            else:
                raise BodoError("pd.Series with dtype: {dtype} not currently supported")

        if no_index:

            def impl(
                data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
            ):  # pragma: no cover
                name_t = bodo.utils.conversion.extract_name_if_none(data, name)
                # Index is empty.
                # TODO: Replace with a valid matching empty index.
                index_t = bodo.hiframes.pd_index_ext.init_range_index(0, 0, 1, None)

                numba.parfors.parfor.init_prange()
                n = len(index_t)
                data_t = bodo.utils.utils.alloc_type(n, _arr_dtype, (-1,))
                return bodo.hiframes.pd_series_ext.init_series(data_t, index_t, name_t)

            return impl

        else:

            def impl(
                data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
            ):  # pragma: no cover
                name_t = bodo.utils.conversion.extract_name_if_none(data, name)
                index_t = bodo.utils.conversion.extract_index_if_none(data, index)

                numba.parfors.parfor.init_prange()
                n = len(index_t)
                data_t = bodo.utils.utils.alloc_type(n, _arr_dtype, (-1,))
                for i in numba.parfors.parfor.internal_prange(n):
                    bodo.libs.array_kernels.setna(data_t, i)

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
    lineterminator=None,
    chunksize=None,
    date_format=None,
    doublequote=True,
    escapechar=None,
    decimal=".",
    errors="strict",
    storage_options=None,
    _bodo_file_prefix="part-",
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
            lineterminator=None,
            chunksize=None,
            date_format=None,
            doublequote=True,
            escapechar=None,
            decimal=".",
            errors="strict",
            storage_options=None,
            _bodo_file_prefix="part-",
            _is_parallel=False,
        ):  # pragma: no cover
            with bodo.ir.object_mode.no_warning_objmode(D="unicode_type"):
                D = series.to_csv(
                    None,
                    sep=sep,
                    na_rep=na_rep,
                    float_format=float_format,
                    columns=columns,
                    header=header,
                    index=index,
                    index_label=index_label,
                    mode=mode,
                    encoding=encoding,
                    compression=compression,
                    quoting=quoting,
                    quotechar=quotechar,
                    lineterminator=lineterminator,
                    chunksize=chunksize,
                    date_format=date_format,
                    doublequote=doublequote,
                    escapechar=escapechar,
                    decimal=decimal,
                    errors=errors,
                    storage_options=storage_options,
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
        lineterminator=None,
        chunksize=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal=".",
        errors="strict",
        storage_options=None,
        _bodo_file_prefix="part-",
        _is_parallel=False,
    ):  # pragma: no cover
        # passing None for the first argument returns a string
        # containing contents to write to csv
        if _is_parallel:
            header &= (bodo.libs.distributed_api.get_rank() == 0) | _csv_output_is_dir(
                unicode_to_utf8(path_or_buf)
            )
        with bodo.ir.object_mode.no_warning_objmode(D="unicode_type"):
            D = series.to_csv(
                None,
                sep=sep,
                na_rep=na_rep,
                float_format=float_format,
                columns=columns,
                header=header,
                index=index,
                index_label=index_label,
                mode=mode,
                encoding=encoding,
                compression=compression,
                quoting=quoting,
                quotechar=quotechar,
                lineterminator=lineterminator,
                chunksize=chunksize,
                date_format=date_format,
                doublequote=doublequote,
                escapechar=escapechar,
                decimal=decimal,
                errors=errors,
                storage_options=storage_options,
            )

        bodo.io.helpers.csv_write(path_or_buf, D, _bodo_file_prefix, _is_parallel)

    return _impl


@lower_constant(SeriesType)
def lower_constant_series(context, builder, series_type, pyval):
    """embed constant Series value by getting constant values for data array and
    Index.
    """
    if isinstance(series_type.data, bodo.types.DatetimeArrayType):
        # TODO [BE-2441]: Unify?
        py_arr = pyval.array
    else:
        py_arr = pyval.values
    data_val = context.get_constant_generic(builder, series_type.data, py_arr)
    index_val = context.get_constant_generic(builder, series_type.index, pyval.index)
    name_val = context.get_constant_generic(builder, series_type.name_typ, pyval.name)

    # create a constant payload with the same data model as SeriesPayloadType
    payload = lir.Constant.literal_struct([data_val, index_val, name_val])
    payload = cgutils.global_constant(builder, ".const.payload", payload).bitcast(
        cgutils.voidptr_t
    )

    # create a constant meminfo with the same data model as Numba
    minus_one = context.get_constant(types.int64, -1)
    null_ptr = context.get_constant_null(types.voidptr)
    meminfo = lir.Constant.literal_struct(
        [minus_one, null_ptr, null_ptr, payload, minus_one]
    )
    meminfo = cgutils.global_constant(builder, ".const.meminfo", meminfo).bitcast(
        cgutils.voidptr_t
    )

    series_val = lir.Constant.literal_struct([meminfo, null_ptr])
    return series_val


# Raise Bodo Error for unsupported attributes and methods of Series
series_unsupported_attrs = {
    # attributes
    "axes",
    "array",  # TODO: support
    "flags",
    "list",
    "struct",
    # Indexing, Iteration
    "at",
    # Computations / descriptive stats
    "is_unique",
    # Accessors
    "sparse",
    # Metadata
    "attrs",
}


series_unsupported_methods = (
    # Axes
    "set_flags",
    # Conversion
    "convert_dtypes",
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
    "case_when",
    # Computations / descriptive stats
    "factorize",
    "mode",
    # Reindexing / selection / label manipulation
    "align",
    "drop",
    "droplevel",
    "reindex",
    "reindex_like",
    "sample",
    "set_axis",
    "truncate",
    "add_prefix",
    "add_suffix",
    "filter",
    # Missing data handling
    "interpolate",
    # Reshaping, sorting
    "reorder_levels",
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
    "to_xarray",
    "to_hdf",
    "to_sql",
    "to_json",
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
        overload_unsupported_attribute(SeriesType, attr_name, full_name)

    for fname in series_unsupported_methods:
        full_name = "Series." + fname
        overload_unsupported_method(SeriesType, fname, full_name)


_install_series_unsupported()


# Raise Bodo Error for unsupported attributes and methods of HeterogenousSeries
heter_series_unsupported_attrs = {
    # attributes
    "axes",
    "array",  # TODO: support
    "dtype",
    "nbytes",
    "memory_usage",
    "hasnans",
    "dtypes",
    "flags",
    # Indexing, Iteration
    "at",
    # Computations / descriptive stats
    "is_unique",
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
    # Accessors
    "dt",
    "str",
    "cat",
    "sparse",
    # Metadata
    "attrs",
}


heter_series_unsupported_methods = {
    # Axes
    "set_flags",
    # Conversion
    "convert_dtypes",
    "infer_objects",
    "copy",
    "bool",
    "to_numpy",
    "to_period",
    "to_timestamp",
    "to_list",
    "tolist",
    "__array__",
    # Indexing, iteration
    "get",
    "at",
    "iat",
    "loc",
    "__iter__",
    "items",
    "iteritems",
    "keys",  # TODO: Support
    "pop",
    "item",
    "xs",
    # Binary operator functions
    "add",
    "sub",
    "mul",
    "div",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "radd",
    "rsub",
    "rmul",
    "rdiv",
    "rtruediv",
    "rfloordiv",
    "rmod",
    "rpow",
    "combine",
    "combine_first",
    "round",
    "lt",
    "gt",
    "le",
    "ge",
    "ne",
    "eq",
    "product",
    "dot",
    # Function application, groupby & window
    "apply",
    "agg",
    "aggregate",
    "transform",
    "map",
    "groupby",
    "rolling",
    "expanding",
    "ewm",
    "pipe",
    # Computations / descriptive stats
    "abs",
    "all",
    "any",
    "autocorr",
    "between",
    "clip",
    "corr",
    "count",
    "cov",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "describe",
    "diff",
    "factorize",
    "kurt",
    "max",
    "mean",
    "median",
    "min",
    "mode",
    "nlargest",
    "nsmallest",
    "pct_change",
    "prod",
    "quantile",
    "rank",
    "sem",
    "skew",
    "std",
    "sum",
    "var",
    "kurtosis",
    "unique",
    "nunique",
    "value_counts",
    # Reindexing / selection / label manipulation
    "align",
    "drop",
    "droplevel",
    "drop_duplicates",
    "duplicated",
    "equals",
    "first",
    "head",
    "idxmax",
    "idxmin",
    "isin",
    "last",
    "reindex",
    "reindex_like",
    "rename",
    "rename_axis",
    "reset_index",
    "sample",
    "set_axis",
    "take",
    "tail",
    "truncate",
    "where",
    "mask",
    "add_prefix",
    "add_suffix",
    "filter",
    # Missing data handling
    "backfill",
    "bfill",
    "dropna",
    "ffill",
    "fillna",
    "interpolate",
    "isna",  # TODO: Support
    "isnull",  # TODO: Support
    "notna",  # TODO: Support
    "notnull",  # TODO: Support
    "pad",
    "replace",
    # Reshaping, sorting
    "argsort",
    "argmin",
    "argmax",
    "reorder_levels",
    "sort_values",
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
    "compare",
    "update",
    # Time series-related
    "asfreq",
    "asof",
    "shift",
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
    "to_csv",
    "to_dict",
    "to_excel",
    "to_frame",
    "to_xarray",
    "to_hdf",
    "to_sql",
    "to_json",
    "to_string",
    "to_clipboard",
    "to_latex",
    "to_markdown",
}


def _install_heter_series_unsupported():
    """install an overload that raises BodoError for unsupported attributes and methods
    of HeterogenousSeries
    """

    for attr_name in heter_series_unsupported_attrs:
        full_name = "HeterogeneousSeries." + attr_name
        overload_unsupported_attribute(HeterogeneousSeriesType, attr_name, full_name)

    for fname in heter_series_unsupported_methods:
        full_name = "HeterogeneousSeries." + fname
        overload_unsupported_method(HeterogeneousSeriesType, fname, full_name)


_install_heter_series_unsupported()
