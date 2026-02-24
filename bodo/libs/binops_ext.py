"""Implementation of binary operators for the different types.
Currently implemented operators:
    arith: add, sub, mul, truediv, floordiv, mod, pow
    cmp: lt, le, eq, ne, ge, gt
"""

import operator

import numba
from numba.core import types
from numba.core.imputils import lower_builtin
from numba.core.typing.builtins import machine_ints
from numba.core.typing.templates import AbstractTemplate, infer_global
from numba.extending import overload

import bodo
from bodo.hiframes.datetime_date_ext import (
    datetime_date_array_type,
    datetime_date_type,
    datetime_timedelta_type,
)
from bodo.hiframes.datetime_timedelta_ext import (
    datetime_datetime_type,
    pd_timedelta_type,
    timedelta_array_type,
)
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_index_ext import (
    DatetimeIndexType,
    HeterogeneousIndexType,
    is_index_type,
)
from bodo.hiframes.pd_offsets_ext import (
    CombinedIntervalType,
    date_offset_type,
    month_begin_type,
    month_end_type,
    week_type,
)
from bodo.hiframes.pd_timestamp_ext import pd_timestamp_tz_naive_type
from bodo.hiframes.series_impl import SeriesType
from bodo.hiframes.time_ext import TimeType
from bodo.hiframes.timestamptz_ext import TimestampTZType
from bodo.libs.binary_arr_ext import binary_array_type, bytes_type
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.decimal_arr_ext import Decimal128Type
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_ext import string_type
from bodo.utils.typing import (
    BodoError,
    is_overload_bool,
    is_overload_none,
    is_str_arr_type,
    is_timedelta_type,
)


class SeriesCmpOpTemplate(AbstractTemplate):
    """
    Split Series Comparison Operators into
    separate Typing and Lowering to Reduce
    Compilation Time.
    """

    def generic(self, args, kws):
        # Builtin operators don't support kws
        # and are always binary
        assert not kws
        assert len(args) == 2
        lhs, rhs = args
        # We only want to call the series implementation if
        # we can't use cmp_timeseries, there are is no dataframe
        # and there is at least 1 Series
        if (
            cmp_timeseries(lhs, rhs)
            or (isinstance(lhs, DataFrameType) or isinstance(rhs, DataFrameType))
            or not (isinstance(lhs, SeriesType) or isinstance(rhs, SeriesType))
        ):
            return

        # Check if we have Timestamp Series with different timezones. If so
        # we can never support this and must throw an exception.
        if is_cmp_tz_mismatch(lhs, rhs):
            # Fetch the timezones for the error message.
            lhs_tz, _ = get_series_tz(lhs)
            rhs_tz, _ = get_series_tz(rhs)
            raise BodoError(
                f"{numba.core.utils.OPERATORS_TO_BUILTINS[self.key]} with two Timestamps requires both Timestamps share the same timezone. "
                + f"Argument 0 has timezone {lhs_tz} and argument 1 has timezone {rhs_tz}. "
                + "To compare these values please convert to timezone naive with ts.tz_convert(None)."
            )

        # Check that lhs and rhs can be legally compared
        # TODO: Replace with a cheaper/more complete check?
        lhs_arr = lhs.data if isinstance(lhs, SeriesType) else lhs
        rhs_arr = rhs.data if isinstance(rhs, SeriesType) else rhs
        # Timestamp and Timestamp need to be unboxed if compared to dt64/td64 array
        if lhs_arr in (
            bodo.types.pd_timestamp_tz_naive_type,
            bodo.types.pd_timedelta_type,
        ) and rhs_arr.dtype in (bodo.types.datetime64ns, bodo.types.timedelta64ns):
            lhs_arr = rhs_arr.dtype
        elif rhs_arr in (
            bodo.types.pd_timestamp_tz_naive_type,
            bodo.types.pd_timedelta_type,
        ) and lhs_arr.dtype in (bodo.types.datetime64ns, bodo.types.timedelta64ns):
            rhs_arr = lhs_arr.dtype

        recursed_args = (lhs_arr, rhs_arr)
        error_msg = f"{lhs} {numba.core.utils.OPERATORS_TO_BUILTINS[self.key]} {rhs} not supported"
        # Check types by resolving the subsequent function
        try:
            ret_arr_typ = self.context.resolve_function_type(
                self.key, recursed_args, {}
            ).return_type
        except Exception:
            raise BodoError(error_msg)
        # If ret_arr_typ is a boolean TRUE for !=, False for ==, then this equality
        # check isn't supported on the array. As a result we raise a BodoError. There may
        # be some cases i.e. S == None where this should be supported,
        # but these are rare.
        if is_overload_bool(ret_arr_typ):
            raise BodoError(error_msg)

        index_typ = lhs.index if isinstance(lhs, SeriesType) else rhs.index
        name_typ = lhs.name_typ if isinstance(lhs, SeriesType) else rhs.name_typ
        # Construct the returned series. The series is always some boolean array
        ret_dtype = types.bool_
        ret_type = SeriesType(ret_dtype, ret_arr_typ, index_typ, name_typ)
        return ret_type(*args)


def series_cmp_op_lower(op):
    def lower_impl(context, builder, sig, args):
        """
        Lowering for Series comparison operators. This should
        be unused unless we are using a compilation path with
        only the Numba pipeline.
        """
        impl = bodo.hiframes.series_impl.create_binary_op_overload(op)(*sig.args)
        # Since this may pick up comparisons that should be rejected, if we don't
        # have a series grab the generic overload.
        # There are two types of inputs that should be rejected:
        #
        # 1 input is a DataFrame:
        #           df == S (although this behavior is deprecated)
        #
        # At least one Series has Timestamp/datetime data:
        #           S(dt64) == S(dt64)
        #
        if impl is None:
            impl = create_overload_cmp_operator(op)(*sig.args)
        return context.compile_internal(builder, impl, sig, args)

    return lower_impl


class SeriesAndOrTyper(AbstractTemplate):
    """
    Operator template used for doing typing for bitwise and logical (And/Or) with pandas series.

    Currently, while pandas allows and/or between non null booleans and integers, we restrict
    and/or to only operate between two boolean types, or two integer types.
    """

    def generic(self, args, kws):
        assert len(args) == 2
        # No kws supported, as builtin operators do not accept them
        assert not kws

        lhs, rhs = args

        if not (isinstance(lhs, SeriesType) or isinstance(rhs, SeriesType)):
            return

        lhs_arr = lhs.data if isinstance(lhs, SeriesType) else lhs
        rhs_arr = rhs.data if isinstance(rhs, SeriesType) else rhs
        recursed_args = (lhs_arr, rhs_arr)
        # TODO: check this error msg is what I want it to be
        error_msg = f"{lhs} {numba.core.utils.OPERATORS_TO_BUILTINS[self.key]} {rhs} not supported"
        try:
            ret_arr_typ = self.context.resolve_function_type(
                self.key, recursed_args, {}
            ).return_type
        except Exception:
            raise BodoError(error_msg)

        index_typ = lhs.index if isinstance(lhs, SeriesType) else rhs.index
        name_typ = lhs.name_typ if isinstance(lhs, SeriesType) else rhs.name_typ
        # Construct the returned series. The series should either be a boolean array,
        # or an integer array
        ret_dtype = ret_arr_typ.dtype
        ret_type = SeriesType(ret_dtype, ret_arr_typ, index_typ, name_typ)
        return ret_type(*args)


def lower_series_and_or(op):
    """
    Returns a lowering implementation for Or/And with pandas series types, to be used with lower_builtin
    """

    def lower_and_or_impl(context, builder, sig, args):  # pragma no cover
        impl = bodo.hiframes.series_impl.create_binary_op_overload(op)(*sig.args)
        # To my understanding, there is only one input type that might be incorrectly matched,
        # and that is
        #
        # 1 input is a DataFrame:
        #           df & S
        #
        if impl is None:
            lhs, rhs = sig.args
            if isinstance(lhs, DataFrameType) or isinstance(rhs, DataFrameType):
                impl = bodo.hiframes.dataframe_impl.create_binary_op_overload(op)(
                    *sig.args
                )

        return context.compile_internal(builder, impl, sig, args)

    return lower_and_or_impl


### Operator.add
def overload_add_operator_scalars(lhs, rhs):
    """Overload types specific to the add operator only.
    Note that the order is important.
    Please don't change unless it's intentional.
    """

    # Offsets
    if lhs == week_type or rhs == week_type:
        return bodo.hiframes.pd_offsets_ext.overload_add_operator_week_offset_type(
            lhs, rhs
        )
    if lhs == month_begin_type or rhs == month_begin_type:
        return (
            bodo.hiframes.pd_offsets_ext.overload_add_operator_month_begin_offset_type(
                lhs, rhs
            )
        )
    if lhs == month_end_type or rhs == month_end_type:
        return bodo.hiframes.pd_offsets_ext.overload_add_operator_month_end_offset_type(
            lhs, rhs
        )
    if lhs == date_offset_type or rhs == date_offset_type:
        return bodo.hiframes.pd_offsets_ext.overload_add_operator_date_offset_type(
            lhs, rhs
        )
    if isinstance(lhs, CombinedIntervalType) or isinstance(rhs, CombinedIntervalType):
        return bodo.hiframes.pd_offsets_ext.overload_add_operator_date_offset_type(
            lhs, rhs
        )

    # The order matters here: make sure offset types are before datetime types
    # Datetime types
    if add_timestamp(lhs, rhs):
        return bodo.hiframes.pd_timestamp_ext.overload_add_operator_timestamp(lhs, rhs)

    if add_dt_td_and_dt_date(lhs, rhs):
        return bodo.hiframes.datetime_date_ext.overload_add_operator_datetime_date(
            lhs, rhs
        )

    if add_datetime_and_timedeltas(lhs, rhs):
        return bodo.hiframes.datetime_timedelta_ext.overload_add_operator_datetime_timedelta(
            lhs, rhs
        )

    raise_error_if_not_numba_supported(operator.add, lhs, rhs)


### Operator.sub
def overload_sub_operator_scalars(lhs, rhs):
    """Overload types specific to the sub operator only.
    Note that the order is important.
    Please don't change unless it's intentional.
    """

    # Offsets
    if sub_offset_to_datetime_or_timestamp(lhs, rhs):
        return bodo.hiframes.pd_offsets_ext.overload_sub_operator_offsets(lhs, rhs)

    # The order matters here: make sure offset types are before datetime types
    # Datetime types
    if isinstance(lhs, bodo.types.PandasTimestampType) and (
        rhs
        in (
            datetime_timedelta_type,
            pd_timedelta_type,
        )
        or isinstance(rhs, bodo.types.PandasTimestampType)
    ):
        # Note we don't support sub between two Timestamps with timezones,
        # but we do with Timestamp and Timedelta
        return bodo.hiframes.pd_timestamp_ext.overload_sub_operator_timestamp(lhs, rhs)

    if sub_dt_or_td(lhs, rhs):
        return bodo.hiframes.datetime_date_ext.overload_sub_operator_datetime_date(
            lhs, rhs
        )

    if sub_datetime_and_timedeltas(lhs, rhs):
        return bodo.hiframes.datetime_timedelta_ext.overload_sub_operator_datetime_timedelta(
            lhs, rhs
        )

    if lhs == datetime_datetime_type and rhs == datetime_datetime_type:
        return (
            bodo.hiframes.datetime_datetime_ext.overload_sub_operator_datetime_datetime(
                lhs, rhs
            )
        )

    raise_error_if_not_numba_supported(operator.sub, lhs, rhs)


## arith operators
def create_overload_arith_op(op):
    """Create overloads for arithmetic operators."""

    def overload_arith_operator(lhs, rhs):
        """Overload some of the arithmetic operators like add, sub, truediv, floordiv, mul, pow, mod."""

        ## ---- start off with redirecting common overloads to some or common to all arith ops:

        # Start of Dataframe Operations
        if isinstance(lhs, DataFrameType) or isinstance(rhs, DataFrameType):
            return bodo.hiframes.dataframe_impl.create_binary_op_overload(op)(lhs, rhs)

        # Start of Series Operations

        # TODO [BE-4027]: Support operator.add with tz-aware series and week offset

        # Time Series for add and sub operators
        if time_series_operation(lhs, rhs) and (op in [operator.add, operator.sub]):
            return bodo.hiframes.series_dt_impl.create_bin_op_overload(op)(lhs, rhs)

        # Generic series
        if isinstance(lhs, SeriesType) or isinstance(rhs, SeriesType):
            return bodo.hiframes.series_impl.create_binary_op_overload(op)(lhs, rhs)

        # Start of Index Operations

        # TODO [BE-4028]: Support operator.add with tz-aware Index and week offset

        # index and timestamp:
        if sub_dt_index_and_timestamp(lhs, rhs) and op == operator.sub:
            return bodo.hiframes.pd_index_ext.overload_sub_operator_datetime_index(
                lhs, rhs
            )

        # Rest of Index Types:
        if operand_is_index(lhs) or operand_is_index(rhs):
            return bodo.hiframes.pd_index_ext.create_binary_op_overload(op)(lhs, rhs)

        # Start of Array Operations

        # non null integer array op pandas timedelta
        if args_td_and_int_array(lhs, rhs):
            return bodo.libs.int_arr_ext.get_int_array_op_pd_td(op)(lhs, rhs)

        # nullable integer array case
        if isinstance(lhs, IntegerArrayType) or isinstance(rhs, IntegerArrayType):
            return bodo.libs.int_arr_ext.create_op_overload(op, 2)(lhs, rhs)

        # nullable float array case
        if isinstance(lhs, FloatingArrayType) or isinstance(rhs, FloatingArrayType):
            return bodo.libs.float_arr_ext.create_op_overload(op, 2)(lhs, rhs)

        # boolean array
        if lhs == boolean_array_type or rhs == boolean_array_type:
            return bodo.libs.bool_arr_ext.create_op_overload(op, 2)(lhs, rhs)

        # String arrays
        if op == operator.add and (
            is_str_arr_type(lhs) or types.unliteral(lhs) == string_type
        ):
            return bodo.libs.str_arr_ext.overload_add_operator_string_array(lhs, rhs)

        # TODO: Timezone Aware Datetime Arrays
        if op == operator.add and (
            isinstance(lhs, bodo.types.DatetimeArrayType)
            or isinstance(rhs, bodo.types.DatetimeArrayType)
        ):
            return bodo.libs.pd_datetime_arr_ext.overload_add_operator_datetime_arr(
                lhs, rhs
            )

        if op == operator.sub and (
            isinstance(lhs, bodo.types.DatetimeArrayType)
            or isinstance(rhs, bodo.types.DatetimeArrayType)
        ):
            return bodo.libs.pd_datetime_arr_ext.overload_sub_operator_datetime_arr(
                lhs, rhs
            )

        # Start of Misc Operations

        # add operator
        if op == operator.add:
            return overload_add_operator_scalars(lhs, rhs)

        # sub operator
        if op == operator.sub:
            return overload_sub_operator_scalars(lhs, rhs)

        # mul operator
        if op == operator.mul:
            # datetime timedelta
            if mul_timedelta_and_int(lhs, rhs):
                return bodo.hiframes.datetime_timedelta_ext.overload_mul_operator_timedelta(
                    lhs, rhs
                )

            # string array
            if mul_string_arr_and_int(lhs, rhs):
                return bodo.libs.str_arr_ext.overload_mul_operator_str_arr(lhs, rhs)

            if mul_date_offset_and_int(lhs, rhs):
                return bodo.hiframes.pd_offsets_ext.overload_mul_date_offset_types(
                    lhs, rhs
                )

            raise_error_if_not_numba_supported(op, lhs, rhs)

        # div operators
        if op in [operator.truediv, operator.floordiv]:
            # pd_timedelta
            if div_timedelta_and_int(lhs, rhs):
                if op == operator.truediv:
                    return bodo.hiframes.datetime_timedelta_ext.overload_truediv_operator_pd_timedelta(
                        lhs, rhs
                    )
                else:
                    return bodo.hiframes.datetime_timedelta_ext.overload_floordiv_operator_pd_timedelta(
                        lhs, rhs
                    )

            # datetime_timedelta
            if div_datetime_timedelta(lhs, rhs):
                if op == operator.truediv:
                    return bodo.hiframes.datetime_timedelta_ext.overload_truediv_operator_dt_timedelta(
                        lhs, rhs
                    )
                else:
                    return bodo.hiframes.datetime_timedelta_ext.overload_floordiv_operator_dt_timedelta(
                        lhs, rhs
                    )

            raise_error_if_not_numba_supported(op, lhs, rhs)

        # mod operator
        if op == operator.mod:
            # timedeltas
            if mod_timedeltas(lhs, rhs):
                return bodo.hiframes.datetime_timedelta_ext.overload_mod_operator_timedeltas(
                    lhs, rhs
                )
            raise_error_if_not_numba_supported(op, lhs, rhs)

        if op == operator.pow:
            raise_error_if_not_numba_supported(op, lhs, rhs)

        # safety net, this stmt should not be reached
        raise BodoError(f"{op} operator not supported for data types {lhs} and {rhs}.")

    return overload_arith_operator


## cmp ops
def create_overload_cmp_operator(op):
    """create overloads for the comparison operators."""

    def overload_cmp_operator(lhs, rhs):
        # Start of Dataframe Operations
        if isinstance(lhs, DataFrameType) or isinstance(rhs, DataFrameType):
            return bodo.hiframes.dataframe_impl.create_binary_op_overload(op)(lhs, rhs)

        # Start of Series Operations

        # time series (order matters: time series check should be before the generic series check)
        if cmp_timeseries(lhs, rhs):
            return bodo.hiframes.series_dt_impl.create_cmp_op_overload(op)(lhs, rhs)

        # series
        if isinstance(lhs, SeriesType) or isinstance(rhs, SeriesType):
            # Use the Series typing template instead.
            return

        # Start of Array Operations
        # Timestamp with timezone
        if isinstance(
            lhs, bodo.libs.pd_datetime_arr_ext.DatetimeArrayType
        ) or isinstance(rhs, bodo.libs.pd_datetime_arr_ext.DatetimeArrayType):
            return bodo.libs.pd_datetime_arr_ext.create_cmp_op_overload_arr(op)(
                lhs, rhs
            )

        # Timezone-naive timestamp array
        if (
            isinstance(lhs, types.Array)
            and lhs.dtype == bodo.types.datetime64ns
            and rhs in (datetime_date_array_type, datetime_date_type)
        ) or (
            lhs in (datetime_date_array_type, datetime_date_type)
            and isinstance(rhs, types.Array)
            and rhs.dtype == bodo.types.datetime64ns
        ):
            return bodo.hiframes.datetime_date_ext.create_datetime_array_date_cmp_op_overload(
                op
            )(lhs, rhs)

        # datetime.date array
        # TODO: this will steal ops from can_cmp_date_datetime case, check if this causes error
        if lhs == datetime_date_array_type or rhs == datetime_date_array_type:
            return bodo.hiframes.datetime_date_ext.create_cmp_op_overload_arr(op)(
                lhs, rhs
            )

        # datetime.timedelta array
        if lhs == timedelta_array_type or rhs == timedelta_array_type:
            impl = bodo.hiframes.datetime_timedelta_ext.create_cmp_op_overload(op)
            return impl(lhs, rhs)

        # str_arr
        if is_str_arr_type(lhs) or is_str_arr_type(rhs):
            return bodo.libs.str_arr_ext.create_binary_op_overload(op)(lhs, rhs)

        # decimal scalar
        if isinstance(lhs, Decimal128Type) or isinstance(rhs, Decimal128Type):
            return bodo.libs.decimal_arr_ext.decimal_create_cmp_op_overload(op)(
                lhs, rhs
            )

        # boolean array
        if lhs == boolean_array_type or rhs == boolean_array_type:
            return bodo.libs.bool_arr_ext.create_op_overload(op, 2)(lhs, rhs)

        # int array
        if isinstance(lhs, IntegerArrayType) or isinstance(rhs, IntegerArrayType):
            return bodo.libs.int_arr_ext.create_op_overload(op, 2)(lhs, rhs)

        # float array
        if isinstance(lhs, FloatingArrayType) or isinstance(rhs, FloatingArrayType):
            return bodo.libs.float_arr_ext.create_op_overload(op, 2)(lhs, rhs)

        if binary_array_cmp(lhs, rhs):
            return bodo.libs.binary_arr_ext.create_binary_cmp_op_overload(op)(lhs, rhs)

        # Datetime Index and String
        if cmp_dt_index_to_string(lhs, rhs):
            return bodo.hiframes.pd_index_ext.overload_binop_dti_str(op)(lhs, rhs)

        # Index Types
        if operand_is_index(lhs) or operand_is_index(rhs):
            return bodo.hiframes.pd_index_ext.create_binary_op_overload(op)(lhs, rhs)

        # datetime.date and datetime64
        if cmp_date_or_datetime64(lhs, rhs):
            return bodo.hiframes.datetime_date_ext.create_cmp_op_overload(op)(lhs, rhs)

        # TimestampTZ
        if isinstance(lhs, TimestampTZType) or isinstance(rhs, TimestampTZType):
            return bodo.hiframes.timestamptz_ext.create_cmp_op_overload(op)(lhs, rhs)

        # time to time comparison
        if isinstance(lhs, TimeType) and isinstance(rhs, TimeType):
            return bodo.hiframes.time_ext.create_cmp_op_overload(op)(lhs, rhs)

        # time to none comparison
        if isinstance(lhs, TimeType) and is_overload_none(rhs):
            return bodo.hiframes.time_ext.create_cmp_op_overload(op)(lhs, rhs)

        if is_overload_none(lhs) and isinstance(rhs, TimeType):
            return bodo.hiframes.time_ext.create_cmp_op_overload(op)(lhs, rhs)

        # datetime.date and datetime.datetime
        if can_cmp_date_datetime(lhs, rhs, op):
            return bodo.hiframes.datetime_date_ext.create_datetime_date_cmp_op_overload(
                op
            )(lhs, rhs)

        # datetime.datetime
        if lhs == datetime_datetime_type and rhs == datetime_datetime_type:
            return bodo.hiframes.datetime_datetime_ext.create_cmp_op_overload(op)(
                lhs, rhs
            )

        # datetime.timedelta
        if lhs == datetime_timedelta_type and rhs == datetime_timedelta_type:
            return bodo.hiframes.datetime_timedelta_ext.create_cmp_op_overload(op)(
                lhs, rhs
            )

        # pd.timedelta
        if cmp_timedeltas(lhs, rhs):
            impl = bodo.hiframes.datetime_timedelta_ext.pd_create_cmp_op_overload(op)
            return impl(lhs, rhs)

        # timestamp
        if cmp_timestamp_or_date(lhs, rhs):
            return bodo.hiframes.pd_timestamp_ext.create_timestamp_cmp_op_overload(op)(
                lhs, rhs
            )

        # if supported by Numba, pass
        if cmp_op_supported_by_numba(lhs, rhs):
            return

        raise BodoError(f"{op} operator not supported for data types {lhs} and {rhs}.")

    return overload_cmp_operator


### Helper Functions For Checking Types


## Helper functions for the add operator
def add_dt_td_and_dt_date(lhs, rhs):
    """Helper function to check types supported in datetime_date_ext overload."""

    lhs_td = lhs == datetime_timedelta_type and rhs == datetime_date_type
    rhs_td = rhs == datetime_timedelta_type and lhs == datetime_date_type
    return lhs_td or rhs_td


def add_timestamp(lhs, rhs):
    """Helper function to check types supported in pd_timestamp_ext overload."""

    ts_and_td = isinstance(lhs, bodo.types.PandasTimestampType) and is_timedelta_type(
        rhs
    )
    td_and_ts = is_timedelta_type(lhs) and isinstance(
        rhs, bodo.types.PandasTimestampType
    )

    return ts_and_td or td_and_ts


def add_datetime_and_timedeltas(lhs, rhs):
    """Helper function to check types supported in datetime_timedelta_ext overload."""

    td_types = [datetime_timedelta_type, pd_timedelta_type]
    deltas = lhs in td_types and rhs in td_types
    dt = (lhs == datetime_datetime_type and rhs in td_types) or (
        rhs == datetime_datetime_type and lhs in td_types
    )

    return deltas or dt


## Helper functions for the mul operator
def mul_string_arr_and_int(lhs, rhs):
    rhs_arr = isinstance(lhs, types.Integer) and is_str_arr_type(rhs)
    lhs_arr = is_str_arr_type(lhs) and isinstance(rhs, types.Integer)

    return rhs_arr or lhs_arr


def mul_timedelta_and_int(lhs, rhs):
    lhs_td = lhs in [pd_timedelta_type, datetime_timedelta_type] and isinstance(
        rhs, types.Integer
    )
    rhs_td = rhs in [pd_timedelta_type, datetime_timedelta_type] and isinstance(
        lhs, types.Integer
    )
    return lhs_td or rhs_td


def mul_date_offset_and_int(lhs, rhs):
    lhs_offset = lhs in [
        week_type,
        month_end_type,
        month_begin_type,
        date_offset_type,
    ] and isinstance(rhs, types.Integer)
    rhs_offset = rhs in [
        week_type,
        month_end_type,
        month_begin_type,
        date_offset_type,
    ] and isinstance(lhs, types.Integer)
    return lhs_offset or rhs_offset


## Helper functions for the sub operator
def sub_offset_to_datetime_or_timestamp(lhs, rhs):
    """Helper function to check types supported in pd_offsets_ext add op overload."""

    tz_naive_dt_types = [
        datetime_datetime_type,
        datetime_date_type,
        pd_timestamp_tz_naive_type,
    ]
    tz_aware_classes = (bodo.types.PandasTimestampType,)
    # offsets
    tz_aware_offset_types = (week_type, month_begin_type, month_end_type)
    tz_naive_offset_types = (date_offset_type,)
    # Here we support all bodo.types.PandasTimestampType only with tz_aware_offset_types and
    # all other types with all offset types.
    return (rhs in tz_aware_offset_types and isinstance(lhs, tz_aware_classes)) or (
        (rhs in tz_naive_offset_types or rhs in tz_aware_offset_types)
        and lhs in tz_naive_dt_types
    )


def sub_dt_index_and_timestamp(lhs, rhs):
    """Helper function to check types supported in pd_index_ext sub op overload."""

    lhs_index = isinstance(lhs, DatetimeIndexType) and rhs == pd_timestamp_tz_naive_type
    rhs_index = isinstance(rhs, DatetimeIndexType) and lhs == pd_timestamp_tz_naive_type

    return lhs_index or rhs_index


def sub_dt_or_td(lhs, rhs):
    """Helper function to check types supported in datetime_date_ext sub op overload."""

    date_and_timedelta = lhs == datetime_date_type and rhs == datetime_timedelta_type
    date_and_date = lhs == datetime_date_type and rhs == datetime_date_type
    date_array_and_timedelta = (
        lhs == datetime_date_array_type and rhs == datetime_timedelta_type
    )

    return date_and_timedelta or date_and_date or date_array_and_timedelta


def sub_datetime_and_timedeltas(lhs, rhs):
    """Helper function to check types supported in datetime_timedelta_ext sub op overload."""

    td_cond = (is_timedelta_type(lhs) or lhs == datetime_datetime_type) and (
        is_timedelta_type(rhs)
    )
    array_cond = (lhs == timedelta_array_type and rhs == datetime_timedelta_type) or (
        lhs == datetime_timedelta_type and rhs == timedelta_array_type
    )

    return td_cond or array_cond


## Helper functions for the div operator
def div_timedelta_and_int(lhs, rhs):
    """Helper function to check types for supported div operator in datetime_timedelta_ext."""

    deltas = lhs == pd_timedelta_type and rhs == pd_timedelta_type
    delta_and_int = lhs == pd_timedelta_type and isinstance(rhs, types.Integer)

    return deltas or delta_and_int


def div_datetime_timedelta(lhs, rhs):
    deltas = lhs == datetime_timedelta_type and rhs == datetime_timedelta_type
    delta_and_int = lhs == datetime_timedelta_type and rhs == types.int64

    return deltas or delta_and_int


## Helper functions for the mod operator
def mod_timedeltas(lhs, rhs):
    pd_deltas = lhs == pd_timedelta_type and rhs == pd_timedelta_type
    dt_deltas = lhs == datetime_timedelta_type and rhs == datetime_timedelta_type

    return pd_deltas or dt_deltas


## Helper functions for the cmp operators
def cmp_dt_index_to_string(lhs, rhs):
    """Helper function to check types supported in pd_index_ext by cmp op overload."""

    lhs_index = (
        isinstance(lhs, DatetimeIndexType) and types.unliteral(rhs) == string_type
    )
    rhs_index = (
        isinstance(rhs, DatetimeIndexType) and types.unliteral(lhs) == string_type
    )

    return lhs_index or rhs_index


def cmp_timestamp_or_date(lhs, rhs):
    """Helper function to check types supported in pd_timestamp_ext by cmp op overload."""

    ts_and_date = (
        isinstance(lhs, bodo.hiframes.pd_timestamp_ext.PandasTimestampType)
        and rhs == bodo.hiframes.datetime_date_ext.datetime_date_type
    )
    date_and_ts = (
        lhs == bodo.hiframes.datetime_date_ext.datetime_date_type
        and isinstance(rhs, bodo.hiframes.pd_timestamp_ext.PandasTimestampType)
    )
    # We will check that timezones match in the implementation.
    ts_and_ts = isinstance(
        lhs, bodo.hiframes.pd_timestamp_ext.PandasTimestampType
    ) and isinstance(rhs, bodo.hiframes.pd_timestamp_ext.PandasTimestampType)

    # Timestamp + dt64
    ts_and_dt64 = (
        isinstance(lhs, bodo.hiframes.pd_timestamp_ext.PandasTimestampType)
        and rhs == bodo.types.datetime64ns
    )
    dt64_and_ts = (
        isinstance(rhs, bodo.hiframes.pd_timestamp_ext.PandasTimestampType)
        and lhs == bodo.types.datetime64ns
    )

    return ts_and_date or date_and_ts or ts_and_ts or ts_and_dt64 or dt64_and_ts


def cmp_date_or_datetime64(lhs, rhs):
    """Helper function to check comparison between datetime.date/datetime64 by cmp op overload."""

    date_and_date = (
        lhs == bodo.hiframes.datetime_date_ext.datetime_date_type
        and rhs == bodo.hiframes.datetime_date_ext.datetime_date_type
    )
    date_and_dt64 = (
        lhs == bodo.hiframes.datetime_date_ext.datetime_date_type
        and rhs == bodo.types.datetime64ns
    )
    dt64_and_date = (
        rhs == bodo.hiframes.datetime_date_ext.datetime_date_type
        and lhs == bodo.types.datetime64ns
    )

    return date_and_date or date_and_dt64 or dt64_and_date


def get_series_tz(val):
    """Get the tz information for a Series, array, or
    scalar value.

    Args:
        val (types.Type): A Bodo type

    Returns:
        Tuple(Optional[str], bool): Returns a tuple of the tz value and if the value
        is a Timestamp value.
    """
    if bodo.hiframes.pd_series_ext.is_dt64_series_typ(val):
        # If we have a Timezone array grab the tz.
        if isinstance(val.data, bodo.libs.pd_datetime_arr_ext.DatetimeArrayType):
            tz = val.data.tz
        else:
            # We are timezone naive.
            tz = None
    elif isinstance(val, bodo.libs.pd_datetime_arr_ext.DatetimeArrayType):
        # We have a Timezone array grab the tz.
        tz = val.tz
    elif isinstance(val, types.Array) and val.dtype == bodo.types.datetime64ns:
        # We are timezone naive.
        tz = None
    elif isinstance(val, bodo.types.PandasTimestampType):
        # If we have a timezone it will be in tz. Naive will be None
        tz = val.tz
    elif val == bodo.types.datetime64ns:
        # We are timezone naive.
        tz = None
    else:
        return None, False
    return tz, True


def is_cmp_tz_mismatch(lhs, rhs):
    """Helper function to determine if two inputs, at least 1 of which is a Series,
    have mismatched timezones. If True will we raise an exception. If False we will
    not. If both arguments are not Timestamps we return False."""
    lhs_tz, lhs_has_tz = get_series_tz(lhs)
    rhs_tz, rhs_has_tz = get_series_tz(rhs)
    # Return if the timezones don't match and both have timezone.
    return lhs_has_tz and rhs_has_tz and lhs_tz != rhs_tz


def cmp_timeseries(lhs, rhs):
    """Helper function to check types supported in series_dt_impl by cmp op overload."""

    dt64s_with_string = bodo.hiframes.pd_series_ext.is_dt64_series_typ(rhs) and (
        bodo.utils.typing.is_overload_constant_str(lhs)
        or lhs == bodo.libs.str_ext.string_type
    )
    string_with_dt64s = bodo.hiframes.pd_series_ext.is_dt64_series_typ(lhs) and (
        bodo.utils.typing.is_overload_constant_str(rhs)
        or rhs == bodo.libs.str_ext.string_type
    )
    is_tz_naive_dt64s = (
        bodo.hiframes.pd_series_ext.is_dt64_series_typ(rhs)
        and rhs.dtype == bodo.types.datetime64ns
        and lhs == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
    ) or (
        bodo.hiframes.pd_series_ext.is_dt64_series_typ(lhs)
        and lhs.dtype == bodo.types.datetime64ns
        and rhs == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
    )
    dt64_series_ops = dt64s_with_string or string_with_dt64s or is_tz_naive_dt64s

    tds_and_td = (
        bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(rhs)
        and lhs == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
    )
    td_and_tds = (
        bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(lhs)
        and rhs == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
    )
    td_series_ops = tds_and_td or td_and_tds

    return dt64_series_ops or td_series_ops


def cmp_timedeltas(lhs, rhs):
    """Helper function to check types supported in datetime_timedelta_ext by cmp op overload."""

    deltas = [pd_timedelta_type, bodo.types.timedelta64ns]
    return lhs in deltas and rhs in deltas


## Generic Helper functions
def operand_is_index(operand):
    return is_index_type(operand) or isinstance(operand, HeterogeneousIndexType)


def helper_time_series_checks(operand):
    """Helper function that checks whether the operand
    type is supported with the dt64_series add/sub ops in series_dt_impl."""
    ret = (
        bodo.hiframes.pd_series_ext.is_dt64_series_typ(operand)
        or bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(operand)
        or operand
        in [datetime_timedelta_type, datetime_datetime_type, pd_timestamp_tz_naive_type]
    )
    return ret


def binary_array_cmp(lhs, rhs):
    """return True if lhs and rhs are both binary array types or one binary array and the other bytes"""
    return (lhs == binary_array_type and rhs in [bytes_type, binary_array_type]) or (
        lhs in [bytes_type, binary_array_type] and rhs == binary_array_type
    )


def can_cmp_date_datetime(lhs, rhs, op):
    """return True if lhs and rhs are a pair of datetime.date and
    datetime.datetime and op is supported by Python"""
    return op in (operator.eq, operator.ne) and (
        (lhs == datetime_date_type and rhs == datetime_datetime_type)
        or (lhs == datetime_datetime_type and rhs == datetime_date_type)
    )


def time_series_operation(lhs, rhs):
    """Helper function to check types supported in series_dt_impl by add/sub op overload."""
    td64series_and_timedelta = (
        bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(lhs)
        and rhs == datetime_timedelta_type
    )
    timedelta_and_td64series = (
        bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(rhs)
        and lhs == datetime_timedelta_type
    )
    dt64series_lhs = bodo.hiframes.pd_series_ext.is_dt64_series_typ(
        lhs
    ) and helper_time_series_checks(rhs)
    dt64series_rhs = bodo.hiframes.pd_series_ext.is_dt64_series_typ(
        rhs
    ) and helper_time_series_checks(lhs)

    return (
        td64series_and_timedelta
        or timedelta_and_td64series
        or dt64series_lhs
        or dt64series_rhs
    )


def args_td_and_int_array(lhs, rhs):
    """helper function to check if the operands consist of a pandas timedelta, and an integer array"""
    one_op_array = (
        isinstance(lhs, IntegerArrayType)
        or (isinstance(lhs, types.Array) and isinstance(lhs.dtype, types.Integer))
    ) or (
        isinstance(rhs, IntegerArrayType)
        or (isinstance(rhs, types.Array) and isinstance(rhs.dtype, types.Integer))
    )
    one_op_pd_td = lhs in [pd_timedelta_type] or rhs in [pd_timedelta_type]
    return one_op_array and one_op_pd_td


## Checks for Numba support
def arith_op_supported_by_numba(op, lhs, rhs):
    """Signatures supported by Numba for binary operators."""

    if op == operator.mul:
        # np timedeltas
        rhs_td = isinstance(lhs, (types.Integer, types.Float)) and isinstance(
            rhs, types.NPTimedelta
        )
        lhs_td = isinstance(rhs, (types.Integer, types.Float)) and isinstance(
            lhs, types.NPTimedelta
        )
        timedeltas = rhs_td or lhs_td

        # unicodes
        rhs_uni = isinstance(rhs, types.UnicodeType) and isinstance(lhs, types.Integer)
        lhs_uni = isinstance(lhs, types.UnicodeType) and isinstance(rhs, types.Integer)
        unicodes = rhs_uni or lhs_uni

        # numbers
        ints = isinstance(lhs, types.Integer) and isinstance(rhs, types.Integer)
        reals = isinstance(lhs, types.Float) and isinstance(rhs, types.Float)
        cmplx = isinstance(lhs, types.Complex) and isinstance(rhs, types.Complex)
        numbers = ints or reals or cmplx

        # Lists
        lists = (isinstance(lhs, types.List) and isinstance(rhs, types.Integer)) or (
            isinstance(lhs, types.Integer) and isinstance(rhs, types.List)
        )

        # char seq
        tys = (types.UnicodeCharSeq, types.CharSeq, types.Bytes)
        char_seq = isinstance(lhs, tys) or isinstance(rhs, tys)

        # arrays
        arrs = isinstance(lhs, types.Array) or isinstance(rhs, types.Array)

        return timedeltas or unicodes or numbers or lists or char_seq or arrs

    if op == operator.pow:
        # int ^ const_int/int
        int_lit = isinstance(lhs, types.Integer) and isinstance(
            rhs, (types.IntegerLiteral, types.Integer)
        )

        # float ^ (float/int/unsigned/signed)
        fl_int = isinstance(lhs, types.Float) and (
            isinstance(
                rhs,
                (types.IntegerLiteral, types.Float, types.Integer)
                or rhs in types.unsigned_domain
                or rhs in types.signed_domain,
            )
        )
        # complex ^ complex
        cmplx = isinstance(lhs, types.Complex) and isinstance(rhs, types.Complex)

        arrs = isinstance(lhs, types.Array) or isinstance(rhs, types.Array)

        return int_lit or fl_int or cmplx or arrs

    if op == operator.floordiv:
        reals = lhs in types.real_domain and rhs in types.real_domain

        ints = isinstance(lhs, types.Integer) and isinstance(rhs, types.Integer)
        floats = isinstance(lhs, types.Float) and isinstance(rhs, types.Float)

        deltas = isinstance(lhs, types.NPTimedelta) and isinstance(
            rhs, (types.Integer, types.Float, types.NPTimedelta)
        )
        arrs = isinstance(lhs, types.Array) or isinstance(rhs, types.Array)

        return reals or ints or floats or deltas or arrs

    if op == operator.truediv:
        mints = lhs in machine_ints and rhs in machine_ints
        reals = lhs in types.real_domain and rhs in types.real_domain
        cmplx = lhs in types.complex_domain and rhs in types.complex_domain

        ints = isinstance(lhs, types.Integer) and isinstance(rhs, types.Integer)
        floats = isinstance(lhs, types.Float) and isinstance(rhs, types.Float)
        complexx = isinstance(lhs, types.Complex) and isinstance(rhs, types.Complex)

        deltas = isinstance(lhs, types.NPTimedelta) and isinstance(
            rhs, (types.Integer, types.Float, types.NPTimedelta)
        )

        arrs = isinstance(lhs, types.Array) or isinstance(rhs, types.Array)

        return mints or reals or cmplx or ints or floats or complexx or deltas or arrs

    if op == operator.mod:
        mints = lhs in machine_ints and rhs in machine_ints
        reals = lhs in types.real_domain and rhs in types.real_domain
        ints = isinstance(lhs, types.Integer) and isinstance(rhs, types.Integer)
        floats = isinstance(lhs, types.Float) and isinstance(rhs, types.Float)

        arrs = isinstance(lhs, types.Array) or isinstance(rhs, types.Array)

        return mints or reals or ints or floats or arrs

    if op == operator.add or op == operator.sub:
        # timedelta
        timedeltas = isinstance(lhs, types.NPTimedelta) and isinstance(
            rhs, types.NPTimedelta
        )

        # NPDatetimes
        dtimes = isinstance(lhs, types.NPDatetime) and isinstance(rhs, types.NPDatetime)

        # NPDatetime and NPTimedelta
        dt_td = isinstance(lhs, types.NPDatetime) and isinstance(rhs, types.NPTimedelta)

        # Sets
        sets = isinstance(lhs, types.Set) and isinstance(rhs, types.Set)

        # Numbers
        ints = isinstance(lhs, types.Integer) and isinstance(rhs, types.Integer)
        reals = isinstance(lhs, types.Float) and isinstance(rhs, types.Float)
        cmplx = isinstance(lhs, types.Complex) and isinstance(rhs, types.Complex)
        numbers = ints or reals or cmplx

        # Arrays
        arrs = isinstance(lhs, types.Array) or isinstance(rhs, types.Array)

        ## add operator only
        # Tuples
        tuples = isinstance(lhs, types.BaseTuple) and isinstance(rhs, types.BaseTuple)

        # Lists
        lists = isinstance(lhs, types.List) and isinstance(rhs, types.List)

        # Chars
        char_seq_char = isinstance(lhs, types.UnicodeCharSeq) and isinstance(
            rhs, types.UnicodeType
        )
        char_char_seq = isinstance(rhs, types.UnicodeCharSeq) and isinstance(
            lhs, types.UnicodeType
        )
        char_seq_char_seq = isinstance(lhs, types.UnicodeCharSeq) and isinstance(
            rhs, types.UnicodeCharSeq
        )
        char_seq_bytes = isinstance(lhs, (types.CharSeq, types.Bytes)) and isinstance(
            rhs, (types.CharSeq, types.Bytes)
        )

        char_add = char_seq_char or char_char_seq or char_seq_char_seq or char_seq_bytes

        # Strings
        unicodes = isinstance(lhs, types.UnicodeType) and isinstance(
            rhs, types.UnicodeType
        )
        char_seq_unicode = isinstance(lhs, types.UnicodeType) and isinstance(
            rhs, types.UnicodeCharSeq
        )

        string_add = unicodes or char_seq_unicode

        # NPTimedelta and NPDatetime
        np_dt = lhs == types.NPTimedelta and rhs == types.NPDatetime

        add_only = tuples or lists or char_add or string_add or np_dt
        add_only_support = op == operator.add and add_only

        return (
            timedeltas or dtimes or dt_td or sets or numbers or arrs or add_only_support
        )


def cmp_op_supported_by_numba(lhs, rhs):
    """Signatures supported by Numba for cmp operator."""

    # arrays
    arrs = isinstance(lhs, types.Array) or isinstance(rhs, types.Array)

    # Lists
    lists = isinstance(lhs, types.ListType) and isinstance(rhs, types.ListType)

    # timedelta
    timedeltas = isinstance(lhs, types.NPTimedelta) and isinstance(
        rhs, types.NPTimedelta
    )

    # datetime.datetime
    datetimes = isinstance(lhs, types.NPDatetime) and isinstance(rhs, types.NPDatetime)

    # unicodes
    unicode_types = (
        types.UnicodeType,
        types.StringLiteral,
        types.CharSeq,
        types.Bytes,
        types.UnicodeCharSeq,
    )
    unicodes = isinstance(lhs, unicode_types) and isinstance(rhs, unicode_types)

    # tuples
    tuples = isinstance(lhs, types.BaseTuple) and isinstance(rhs, types.BaseTuple)

    # sets
    sets = isinstance(lhs, types.Set) and isinstance(rhs, types.Set)

    # numbers
    numbers = isinstance(lhs, types.Number) and isinstance(rhs, types.Number)

    # bools
    bools = isinstance(lhs, types.Boolean) and isinstance(rhs, types.Boolean)

    # None
    nones = isinstance(lhs, types.NoneType) or isinstance(rhs, types.NoneType)

    # dictionaries
    dicts = isinstance(lhs, types.DictType) and isinstance(rhs, types.DictType)

    # enums
    enums = isinstance(lhs, types.EnumMember) and isinstance(rhs, types.EnumMember)

    # literals
    literals = isinstance(lhs, types.Literal) and isinstance(rhs, types.Literal)

    return (
        lists
        or timedeltas
        or datetimes
        or unicodes
        or tuples
        or sets
        or numbers
        or bools
        or nones
        or dicts
        or arrs
        or enums
        or literals
    )


## Helper function for raising errors
def raise_error_if_not_numba_supported(op, lhs, rhs):
    """If arithmetic operator supported by Numba pass, otherwise raise a BodoError."""

    if arith_op_supported_by_numba(op, lhs, rhs):
        return

    raise BodoError(f"{op} operator not supported for data types {lhs} and {rhs}.")


def _install_series_and_or():
    """Installs the overloads for series and/or operations"""
    for op in (operator.or_, operator.and_):
        infer_global(op)(SeriesAndOrTyper)
        lower_impl = lower_series_and_or(op)
        lower_builtin(op, SeriesType, SeriesType)(lower_impl)
        lower_builtin(op, SeriesType, types.Any)(lower_impl)
        lower_builtin(op, types.Any, SeriesType)(lower_impl)


_install_series_and_or()


## Install operator overloads
def _install_cmp_ops():
    for op in (
        operator.lt,
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
        operator.le,
    ):
        # Install Series typing
        infer_global(op)(SeriesCmpOpTemplate)
        # Install the series lowering
        # TODO: Update the lower builtin to be more accurate. We want
        # to match on any implementation handled by Bodo and not Numba.
        lower_impl = series_cmp_op_lower(op)
        lower_builtin(op, SeriesType, SeriesType)(lower_impl)
        lower_builtin(op, SeriesType, types.Any)(lower_impl)
        lower_builtin(op, types.Any, SeriesType)(lower_impl)
        # Include the generic overload
        overload_impl = create_overload_cmp_operator(op)
        overload(op, no_unliteral=True)(overload_impl)


_install_cmp_ops()


def install_arith_ops():
    """Install arithmetic operators overload."""

    for op in (
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.floordiv,
        operator.mod,
        operator.pow,
    ):
        overload_impl = create_overload_arith_op(op)
        overload(op, no_unliteral=True)(overload_impl)


install_arith_ops()
