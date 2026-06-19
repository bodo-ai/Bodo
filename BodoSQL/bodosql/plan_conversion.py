from __future__ import annotations

import decimal
import operator
import re
import zoneinfo
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import py4j
import pyarrow as pa
import pyarrow.compute as pc

import bodo
import bodo.pandas as bd
import bodosql
from bodo.pandas.plan import (
    AggregateExpression,
    ArithOpExpression,
    ArrowScalarFuncExpression,
    CaseExpression,
    CastExpression,
    ColRefExpression,
    ComparisonOpExpression,
    ConjunctionOpExpression,
    ConstantExpression,
    LazyPlan,
    LogicalAggregate,
    LogicalComparisonJoin,
    LogicalCrossProduct,
    LogicalDistinct,
    LogicalFilter,
    LogicalJoinFilter,
    LogicalOrder,
    LogicalProjection,
    LogicalTopN,
    NullExpression,
    UnaryOpExpression,
    arrow_to_empty_df,
    make_col_ref_exprs,
)
from bodo.pandas.utils import wrap_plan
from bodosql.imported_java_classes import JavaEntryPoint, gateway

_DATE_PART_ARROW_FUNCS = {
    "YEAR": "year",
    "MONTH": "month",
    "DAY": "day",
    "DAYOFMONTH": "day",
    "HOUR": "hour",
    "MINUTE": "minute",
    "SECOND": "second",
    "QUARTER": "quarter",
    "WEEK": "iso_week",
    "WEEKOFYEAR": "iso_week",
    "WEEKISO": "iso_week",
    "DAYOFYEAR": "day_of_year",
    "DAYOFWEEK": "day_of_week",
    "WEEKDAY": "day_of_week",
    "MICROSECOND": "microsecond",
    "NANOSECOND": "nanosecond",
    "DOW": "day_of_week",
    "DOY": "day_of_year",
}
INTERVAL_UNIT_MAP = {
    "YEAR": ("year", lambda n: (12 * n, 0, 0)),
    "QUARTER": ("quarter", lambda n: (3 * n, 0, 0)),
    "MONTH": ("month", lambda n: (n, 0, 0)),
    "WEEK": ("week", lambda n: (0, 7 * n, 0)),
    "DAY": ("day", lambda n: (0, n, 0)),
    "HOUR": ("hour", lambda n: (0, 0, 3600 * n * 1_000_000_000)),
    "MINUTE": ("minute", lambda n: (0, 0, 60 * n * 1_000_000_000)),
    "SECOND": ("second", lambda n: (0, 0, n * 1_000_000_000)),
    "MS": ("millisecond", lambda n: (0, 0, n * 1_000_000)),
    "MILLISECOND": ("millisecond", lambda n: (0, 0, n * 1_000_000)),
    "MICROSECOND": ("microsecond", lambda n: (0, 0, n * 1_000)),
    "NANOSECOND": ("nanosecond", lambda n: (0, 0, n)),
}
_MYSQL_TO_STRFTIME = {
    "%a": "%a",
    "%b": "%b",
    "%d": "%d",
    "%H": "%H",
    "%h": "%I",
    "%I": "%I",
    "%i": "%M",
    "%j": "%j",
    "%M": "%B",
    "%m": "%m",
    "%p": "%p",
    "%r": "%H:%M:%OS %p",
    "%T": "%H:%M:%OS",
    "%s": "%OS",
    "%S": "%OS",
    "%U": "%U",
    "%u": "%W",
    "%W": "%A",
    "%w": "%w",
    "%Y": "%Y",
    "%y": "%y",
    "%%": "%%",
}
_MYSQL_FORMAT_TOKEN_RE = re.compile(r"%(.)|%$")


def _mysql_date_format_to_arrow_format(mysql_fmt: str) -> str:
    def replace_mysql_token(match):
        mysql_token = match.group(0)
        if mysql_token == "%f":
            raise NotImplementedError(
                "DATE_FORMAT with '%f' (microseconds) is not supported in the C++ backend yet "
                "because PyArrow's strftime does not handle it correctly."
            )
        if mysql_token == "%":
            return "%%"
        return _MYSQL_TO_STRFTIME.get(mysql_token, mysql_token[1])

    return _MYSQL_FORMAT_TOKEN_RE.sub(replace_mysql_token, mysql_fmt)


@dataclass
class IcebergReadInfo:
    """Information extracted from Iceberg read plan nodes."""

    scan_node: object = None
    filters: list[object] = None
    # Columns to read from the table, in the order they should appear in output.
    colmap: list[int] = None
    limit: int = None


def java_plan_to_python_plan(ctx, java_plan):
    """Convert a BodoSQL Java plan (RelNode) to a DataFrame library plan
    (bodo.pandas.plan.LazyPlan) for execution in the C++ runtime backend.
    """
    java_class_name = java_plan.getClass().getSimpleName()

    if java_class_name in (
        "PandasToBodoPhysicalConverter",
        "CombineStreamsExchange",
        "SeparateStreamExchange",
    ):
        # PandasToBodoPhysicalConverter is a no-op
        # CombineStreamsExchange is a no-op here since C++ runtime accumulates results
        # in output buffer by default
        # SeparateStreamExchange is a no-op here since PhysicalReadPandas in C++ runtime
        # streams data in batches by default
        input = java_plan.getInput()
        return java_plan_to_python_plan(ctx, input)

    if java_class_name == "PandasTableScan":
        # TODO: support other table types and check table details
        table_name = JavaEntryPoint.getLocalTableName(java_plan)
        table = ctx.tables[table_name]
        if isinstance(table, bodosql.TablePath):
            if table._file_type == "pq":
                return bd.read_parquet(table._file_path)._plan
            else:
                raise NotImplementedError(
                    f"TablePath with file type {table._file_type} not supported in C++ backend yet"
                )
        elif isinstance(table, bodo.pandas.BodoDataFrame):
            return table._plan
        elif isinstance(table, pd.DataFrame):
            return bodo.pandas.from_pandas(table)._plan
        else:
            raise NotImplementedError(
                f"Table type {type(table)} not supported in C++ backend yet"
            )

    # Traverse Iceberg plan nodes to extract read information similar to BodoSQL
    # (see flattenIcebergTree in BodoSQL Java code)
    if java_class_name == "IcebergToBodoPhysicalConverter":
        input = java_plan.getInput()
        read_info = IcebergReadInfo()
        # Initialize all columns to be in the original location (updated top-down based
        # on IcebergProject nodes)
        read_info.colmap = list(range(input.getRowType().getFieldCount()))
        visit_iceberg_node(input, read_info)
        return generate_iceberg_read(read_info)

    if java_class_name in ("PandasProject", "BodoPhysicalProject"):
        input_plan = java_plan_to_python_plan(ctx, java_plan.getInput())
        exprs = [
            java_expr_to_python_expr(ctx, e, input_plan)
            for e in java_plan.getProjects()
        ]
        names = list(java_plan.getRowType().getFieldNames())
        new_schema = pa.schema(
            [pa.field(name, e.pa_schema.field(0).type) for e, name in zip(exprs, names)]
        )
        empty_data = arrow_to_empty_df(new_schema)
        proj_plan = LogicalProjection(
            empty_data,
            input_plan,
            exprs,
        )
        return proj_plan

    if java_class_name == "BodoPhysicalJoin":
        return java_join_to_python_join(ctx, java_plan)

    if java_class_name == "BodoPhysicalRuntimeJoinFilter":
        return java_rtjf_to_python_rtjf(ctx, java_plan)

    if java_class_name == "BodoPhysicalFilter":
        return java_filter_to_python_filter(ctx, java_plan)

    if java_class_name == "BodoPhysicalAggregate" and not java_plan.usesGroupingSets():
        # TODO: support grouping sets
        return java_agg_to_python_agg(ctx, java_plan)

    if java_class_name == "BodoPhysicalSort":
        return java_sort_to_python_sort(ctx, java_plan)

    if java_class_name == "BodoPhysicalValues":
        return java_values_to_python_values(ctx, java_plan)

    if java_class_name == "BodoPhysicalCachedSubPlan":
        return java_subplan_to_python_subplan(ctx, java_plan)

    raise NotImplementedError(f"Plan node {java_class_name} not supported yet")


def java_expr_to_python_expr(ctx, java_expr, input_plan):
    """Convert a BodoSQL Java expression to a DataFrame library expression
    (bodo.pandas.plan.Expression).
    """
    java_class_name = java_expr.getClass().getSimpleName()

    if java_class_name == "RexInputRef":
        col_index = java_expr.getIndex()
        return make_col_ref_exprs([col_index], input_plan)[0]

    if java_class_name == "RexCall":
        return java_call_to_python_call(ctx, java_expr, input_plan)

    if java_class_name == "RexLiteral":
        return java_literal_to_python_literal(ctx, java_expr, input_plan)

    raise NotImplementedError(f"Expression {java_class_name} not supported yet")


def java_call_to_python_call(ctx, java_call, input_plan):
    """Convert a BodoSQL Java call expression to a DataFrame library expression
    (bodo.pandas.plan.Expression).
    """
    op = java_call.getOperator()
    operator_class_name = op.getClass().getSimpleName()

    SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind

    _DOW_NAMES = {"DAYOFWEEK", "DOW"}

    if operator_class_name == "SqlNullPolicyFunction":
        func_name = op.getName().upper()
        num_operands = len(java_call.getOperands())

        # Date part functions wrapped in SqlNullPolicyFunction (e.g. WEEKDAY($0))
        if func_name in _DATE_PART_ARROW_FUNCS and num_operands == 1:
            input = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            arrow_func = _DATE_PART_ARROW_FUNCS[func_name]
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            raw_expr = ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())
            if func_name in ("DAYOFWEEK", "DOW"):
                # PyArrow default: 0=Monday, 6=Sunday.
                # Bodo/Snowflake DAYOFWEEK: 1=Monday, 0=Sunday → (dow+1)%7
                one = ConstantExpression(empty_data, input_plan, 1)
                seven = ConstantExpression(empty_data, input_plan, 7)
                raw_expr = ArithOpExpression(empty_data, raw_expr, one, "__add__")
                raw_expr = ArithOpExpression(empty_data, raw_expr, seven, "__mod__")
            # For WEEKDAY, PyArrow default (0=Monday..6=Sunday) exactly matches
            # Spark/Snowflake WEEKDAY, so no transformation is needed.
            return raw_expr

        if func_name == "DAYNAME" and num_operands == 1:
            input = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.string()))
            return ArrowScalarFuncExpression(empty_data, [input], "strftime", ("%a",))

        if func_name in ("MONTHNAME", "MONTH_NAME") and num_operands == 1:
            input = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.string()))
            return ArrowScalarFuncExpression(empty_data, [input], "strftime", ("%b",))

        if func_name == "DATE_FORMAT" and num_operands == 2:
            input = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            mysql_fmt = str(java_call.getOperands()[1].toString())
            if mysql_fmt.startswith("'") and mysql_fmt.endswith("'"):
                mysql_fmt = mysql_fmt[1:-1]
            arrow_fmt = _mysql_date_format_to_arrow_format(mysql_fmt)
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.string()))
            return ArrowScalarFuncExpression(
                empty_data, [input], "strftime", (arrow_fmt,)
            )

        if func_name == "MAKEDATE" and num_operands == 2:
            # MAKEDATE(year, dayofyear) → Jan 1 of year + (doy-1) days
            year_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            doy_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[1], input_plan
            )
            assert isinstance(year_expr, ConstantExpression) and isinstance(
                doy_expr, ConstantExpression
            ), "MAKEDATE requires constant integer arguments in C++ backend"
            result_date = datetime(int(year_expr.value), 1, 1).date() + timedelta(
                days=int(doy_expr.value) - 1
            )
            empty_data = pd.Series([result_date], dtype=pd.ArrowDtype(pa.date32()))
            return ConstantExpression(empty_data, input_plan, result_date)

        if func_name == "DATE_TRUNC" and num_operands == 2:
            # DATE_TRUNC(FLAG(DAY), timestamp) → floor_temporal(timestamp, unit)
            unit_raw = str(java_call.getOperands()[0].toString()).upper()
            if "(" in unit_raw:
                unit_raw = unit_raw.split("(")[1].rstrip(")")
            _TRUNC_UNITS = {
                "year",
                "quarter",
                "month",
                "week",
                "day",
                "hour",
                "minute",
                "second",
            }
            assert unit_raw.lower() in _TRUNC_UNITS, (
                f"DATE_TRUNC unit has unexpected format after stripping: "
                f"'{unit_raw}' (original: "
                f"'{java_call.getOperands()[0].toString()}')"
            )
            arrow_unit = unit_raw.lower()
            input = java_expr_to_python_expr(
                ctx, java_call.getOperands()[1], input_plan
            )
            input_type = input.pa_schema.field(0).type
            if pa.types.is_time(input_type) and arrow_unit in (
                "year",
                "quarter",
                "month",
                "week",
                "day",
            ):
                raise NotImplementedError(
                    f"Unsupported unit for DATE_TRUNC with TIME input: {unit_raw}"
                )
            empty_data = pd.Series(
                [],
                dtype=pd.ArrowDtype(
                    sql_type_to_pa_type(ctx, java_call.getType().getSqlTypeName())
                ),
            )
            return ArrowScalarFuncExpression(
                empty_data, [input], "floor_temporal", (1, arrow_unit)
            )

        if func_name == "LAST_DAY":
            date_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            unit_str = "MONTH"
            if num_operands == 2:
                # EXTRACT(FLAG(MONTH), date) → month
                unit_str = str(java_call.getOperands()[1].toString()).upper()
                # Strip "FLAG(" / ")" from unit string
                if "(" in unit_str:
                    unit_str = unit_str.split("(")[1].rstrip(")")
            LAST_DAY_UNITS = {
                "MONTH": ("month", 1, 0),
                "QUARTER": ("quarter", 3, 0),
                "YEAR": ("year", 12, 0),
                "WEEK": ("week", 0, 7),
            }
            assert unit_str in LAST_DAY_UNITS, f"Unsupported LAST_DAY unit: {unit_str}"
            empty_data = date_expr.empty_data

            # Arrow doesn't have last day built in so emulate with date_trunc + interval + date arithmetic
            # Maps unit -> (floor_temporal_unit, MonthDayNano months, MonthDayNano days)
            # Note: WEEK uses days=7; others use months; all subtract 1 day at the end

            floor_unit, interval_months, interval_days = LAST_DAY_UNITS[unit_str]

            truncated = ArrowScalarFuncExpression(
                empty_data, [date_expr], "floor_temporal", (1, floor_unit)
            )
            next_period = ArithOpExpression(
                empty_data,
                truncated,
                ConstantExpression(
                    empty_data,
                    input_plan,
                    ("MonthDayNanoInterval", interval_months, interval_days, 0),
                ),
                "__add__",
            )
            last_day = ArithOpExpression(
                empty_data,
                next_period,
                ConstantExpression(
                    empty_data, input_plan, ("MonthDayNanoInterval", 0, 1, 0)
                ),
                "__sub__",
            )
            return last_day

        # ADD_MONTHS(date, months) -> date + months * INTERVAL '1' MONTH
        if func_name == "ADD_MONTHS" and num_operands == 2:
            date_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            months_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[1], input_plan
            )
            assert isinstance(months_expr, ConstantExpression) and isinstance(
                months_expr.value, int
            ), "ADD_MONTHS requires constant integer for months argument in C++ backend"
            month_interval = ("MonthDayNanoInterval", months_expr.value, 0, 0)
            dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.duration("ns")))
            month_interval_expr = ConstantExpression(
                dummy_empty_data, input_plan, month_interval
            )

            # Add to date (output type is same as input date type)
            return ArithOpExpression(
                date_expr.empty_data, date_expr, month_interval_expr, "__add__"
            )

        if func_name in ("DATEADD", "DATE_ADD", "ADDDATE", "TIMEADD", "TIMESTAMPADD"):
            # DATEADD(date, interval) or DATEADD(unit, amount, date)
            # For 2 operands: (date, interval) → date + interval
            # For 3 operands: (unit, amount, date) → date + (unit * amount)
            if num_operands == 2:
                # 2-operand form: DATEADD(date, interval) → date + interval
                # This path handles Snowflake-style DATEADD with a date/timestamp
                # and an interval literal. Default to DAY-based units
                # (86_400_000_000_000 nanos per day) for the scalar fallback.
                date_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[0], input_plan
                )
                amount_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[1], input_plan
                )
                int_empty = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
                out_empty = pd.Series(dtype=pd.ArrowDtype(pa.timestamp("ns")))
                return ArrowScalarFuncExpression(
                    out_empty,
                    [
                        date_expr,
                        amount_expr,
                        ConstantExpression(int_empty, input_plan, 0),
                        ConstantExpression(int_empty, input_plan, 86_400_000_000_000),
                    ],
                    "bodo_dateadd",
                    (),
                )
            elif num_operands == 3:
                # 3-operand form: DATEADD(unit, amount, date) → date + (unit * amount)
                # First operand is a FLAG(unit) interval qualifier from the Java
                # planner (e.g. FLAG(DAY), FLAG(MONTH)).
                first_op_str = str(java_call.getOperands()[0].toString())
                if not first_op_str.startswith("FLAG"):
                    raise NotImplementedError(
                        "DATEADD with 3 string operands not supported in "
                        "C++ backend yet"
                    )
                unit_str = first_op_str.split("(")[1].rstrip(")").upper()
                assert unit_str in INTERVAL_UNIT_MAP, (
                    f"Unsupported DATEADD interval unit: {unit_str}"
                )
                amount_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[1], input_plan
                )
                date_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[2], input_plan
                )
                # Decompose the interval unit into (months, days, nanos) for
                # per-unit values. E.g. YEAR → (12, 0, 0), DAY → (0, 1, 0).
                interval_months, interval_days, interval_nanos = INTERVAL_UNIT_MAP[
                    unit_str
                ][1](1)
                date_pa_type = date_expr.empty_data.iloc[:, 0].dtype.pyarrow_dtype
                is_date_input = pa.types.is_date(date_pa_type)
                # Preserve the date output type when the input is a date (not
                # timestamp) and the interval has only month/day components
                # (no sub-day precision). E.g. DATEADD(DAY, 5, date_col)
                # returns a date.
                preserves_date_type = is_date_input and interval_nanos == 0
                out_empty = pd.Series(
                    dtype=date_expr.empty_data.iloc[:, 0].dtype
                    if preserves_date_type
                    else pd.ArrowDtype(pa.time64("ns"))
                    if pa.types.is_time64(date_pa_type)
                    else pd.ArrowDtype(pa.timestamp("ns"))
                )

                # Special case: tz-aware timestamps with month-based intervals
                # (YEAR/QUARTER/MONTH). The C++ bodo_dateadd scalar function
                # doesn't handle tz-aware timestamps with month intervals,
                # so fall back to ArrowArithOpExpression which delegates to
                # Arrow's month interval arithmetic. Only works for constant
                # amounts since Arrow requires a literal month interval.
                if (
                    interval_months != 0
                    and pa.types.is_timestamp(date_pa_type)
                    and date_pa_type.tz is not None
                    and isinstance(amount_expr, ConstantExpression)
                ):
                    # MonthDayNanoInterval tuple: (unit, months, days, nanos).
                    # Days and nanos are always zero here because this branch
                    # only runs for month-based units (YEAR/QUARTER/MONTH) which
                    # only produce non-zero months component from INTERVAL_UNIT_MAP.
                    month_interval = (
                        "MonthDayNanoInterval",
                        interval_months
                        * int(
                            amount_expr.value
                            + (0.5 if amount_expr.value >= 0 else -0.5)
                        ),
                        0,  # days: always 0 for month-based units
                        0,  # nanos: always 0 for month-based units
                    )
                    dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.duration("ns")))
                    month_interval_expr = ConstantExpression(
                        dummy_empty_data, input_plan, month_interval
                    )
                    return ArithOpExpression(
                        out_empty, date_expr, month_interval_expr, "__add__"
                    )

                # General path: convert the interval unit to nanos-per-unit
                # and pass (months, nanos_per_unit) to bodo_dateadd.
                # nanos_per_unit is zero for month-based units and non-zero
                # for day/time-based units.
                int_empty = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
                nanos_per_unit = interval_days * 86_400_000_000_000 + interval_nanos
                return ArrowScalarFuncExpression(
                    out_empty,
                    [
                        date_expr,
                        amount_expr,
                        ConstantExpression(int_empty, input_plan, interval_months),
                        ConstantExpression(int_empty, input_plan, nanos_per_unit),
                    ],
                    "bodo_dateadd",
                    (),
                )
        if func_name in ("DATE_SUB", "SUBDATE"):
            # DATE_SUB(date, interval) or DATE_SUB(unit, amount, date)
            # or DATE_SUB(date, integer_days) — Snowflake syntax.
            if num_operands == 2:
                date_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[0], input_plan
                )
                amount_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[1], input_plan
                )
                # Check if the second argument is an integer (number of days)
                # rather than an INTERVAL literal.
                amount_type = java_call.getOperands()[1].getType()
                SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName
                if is_int_type(amount_type):
                    # DATE_SUB(date, N) → date - N days
                    if isinstance(amount_expr, ConstantExpression):
                        interval_val = pd.Timedelta(days=int(amount_expr.value))
                        dummy_empty_data = pd.Series(
                            dtype=pd.ArrowDtype(pa.duration("ns"))
                        )
                        interval_expr = ConstantExpression(
                            dummy_empty_data, input_plan, interval_val
                        )
                    else:
                        one_day = pd.Timedelta(days=1)
                        dummy_empty_data = pd.Series(
                            dtype=pd.ArrowDtype(pa.duration("ns"))
                        )
                        one_day_expr = ConstantExpression(
                            dummy_empty_data, input_plan, one_day
                        )
                        interval_expr = ArithOpExpression(
                            dummy_empty_data,
                            amount_expr,
                            one_day_expr,
                            "__mul__",
                        )
                    out_empty = date_expr.empty_data - interval_expr.empty_data
                    return ArithOpExpression(
                        out_empty, date_expr, interval_expr, "__sub__"
                    )
                # Otherwise DATE_SUB(date, interval) — direct subtraction.
                return java_binop_to_python_expr(
                    ctx,
                    SqlKind.MINUS,
                    "-",
                    [date_expr, amount_expr],
                )
            elif num_operands == 3:
                return java_binop_to_python_expr(
                    ctx,
                    SqlKind.MINUS,
                    "-",
                    [
                        java_expr_to_python_expr(
                            ctx, java_call.getOperands()[2], input_plan
                        ),
                        java_expr_to_python_expr(
                            ctx, java_call.getOperands()[1], input_plan
                        ),
                    ],
                )
        if func_name == "TIME_SLICE":
            # TIME_SLICE(date, interval) → floor_temporal(date, interval)
            date_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            interval_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[1], input_plan
            )
            unit_str = str(java_call.getOperands()[2].toString()).upper().strip("'")
            # Strip "FLAG(" / ")" from unit string
            if "(" in unit_str:
                unit_str = unit_str.split("(")[1].rstrip(")")
            start_or_end = "START"
            if num_operands == 4:
                start_or_end = (
                    str(java_call.getOperands()[3].toString()).upper().strip("'")
                )
                assert start_or_end in ("START", "END"), (
                    f"Unsupported TIME_SLICE 4th operand: {start_or_end}"
                )
            assert isinstance(interval_expr, ConstantExpression), (
                "TIME_SLICE interval must be a constant in C++ backend"
            )
            slice_length = int(interval_expr.value)
            assert unit_str in INTERVAL_UNIT_MAP, (
                f"Unsupported TIME_SLICE interval unit: {unit_str}"
            )
            empty_data = date_expr.empty_data

            arrow_unit, interval_fn = INTERVAL_UNIT_MAP[unit_str]
            interval_months, interval_days, interval_nanos = interval_fn(slice_length)

            truncated = ArrowScalarFuncExpression(
                empty_data, [date_expr], "floor_temporal", (slice_length, arrow_unit)
            )

            if start_or_end == "START":
                return truncated

            # END: return start of next slice
            return ArithOpExpression(
                empty_data,
                truncated,
                ConstantExpression(
                    empty_data,
                    input_plan,
                    (
                        "MonthDayNanoInterval",
                        interval_months,
                        interval_days,
                        interval_nanos,
                    ),
                ),
                "__add__",
            )

    if operator_class_name in (
        "SqlMonotonicBinaryOperator",
        "SqlBinaryOperator",
        "SqlDatetimePlusOperator",
        "SqlDatetimeSubtractionOperator",
    ):
        operands = java_call.getOperands()
        # Calcite may add more than 2 operand for the same binary operator
        op_exprs = [java_expr_to_python_expr(ctx, o, input_plan) for o in operands]
        kind = op.getKind()
        return java_binop_to_python_expr(ctx, kind, op.getName(), op_exprs)

    if operator_class_name == "SqlCastFunction" and len(java_call.getOperands()) == 1:
        operand = java_call.getOperands()[0]
        operand_type = operand.getType()
        target_type = java_call.getType()
        SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName
        in_expr = java_expr_to_python_expr(ctx, operand, input_plan)
        # TODO[BSE-5154]: support all Calcite casts

        # No-op casts
        if operand_type.getSqlTypeName().equals(target_type.getSqlTypeName()):
            return in_expr

        if target_type.getSqlTypeName().equals(SqlTypeName.DECIMAL) and is_int_type(
            operand_type
        ):
            # Cast of int to DECIMAL is unnecessary in C++ backend
            return in_expr

        if operand_type.getSqlTypeName().equals(
            SqlTypeName.VARCHAR
        ) and target_type.getSqlTypeName().equals(SqlTypeName.VARCHAR):
            # No-op cast of VARCHAR (could be different lengths but sometimes equal
            # which seems like a Calcite gap)
            return in_expr

        if operand_type.getSqlTypeName().equals(
            SqlTypeName.DATE
        ) and target_type.getSqlTypeName().equals(SqlTypeName.TIMESTAMP):
            # Cast of DATE to TIMESTAMP is unnecessary in C++ backend
            return in_expr

        empty_data = pd.Series(
            dtype=pd.ArrowDtype(sql_type_to_pa_type(ctx, target_type.getSqlTypeName()))
        )

        # TO_TIMESTAMP/TO_TIMESTAMP_NTZ remove the timezone which is same as
        # local_timestamp() function of Arrow (not cast)
        if operand_type.getSqlTypeName().equals(
            SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE
        ) and target_type.getSqlTypeName().equals(SqlTypeName.TIMESTAMP):
            return ArrowScalarFuncExpression(
                empty_data,
                [in_expr],
                "local_timestamp",
                (),
            )

        # TO_TIMESTAMP_LTZ adds local time zone which is same as assume_timezone()
        # function of Arrow (not cast)
        if target_type.getSqlTypeName().equals(
            SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE
        ):
            if not operand_type.getSqlTypeName().equals(SqlTypeName.TIMESTAMP):
                # Integers are assumed in seconds in BodoSQL
                cast_empty_data = pd.Series(
                    dtype=pd.ArrowDtype(
                        pa.timestamp("s" if is_int_type(operand_type) else "ns")
                    )
                )
                in_expr = CastExpression(
                    cast_empty_data,
                    in_expr,
                )

            # BodoSQL uses UTC if timezone is not specified
            tz = ctx.default_tz if ctx.default_tz is not None else "UTC"
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.timestamp("ns", tz=tz)))
            return ArrowScalarFuncExpression(
                empty_data,
                [in_expr],
                "assume_timezone",
                (tz,),
            )

        # Integers are assumed in seconds in BodoSQL
        if is_int_type(operand_type) and target_type.getSqlTypeName().equals(
            SqlTypeName.TIMESTAMP
        ):
            cast_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.timestamp("s")))
            in_expr = CastExpression(
                cast_empty_data,
                in_expr,
            )

        # Parsing strings to binary not supported yet
        if operand_type.getSqlTypeName().equals(
            SqlTypeName.VARCHAR
        ) and target_type.getSqlTypeName().equals(SqlTypeName.VARBINARY):
            raise NotImplementedError(
                "Cast of VARCHAR to VARBINARY is not supported in C++ backend yet"
            )

        return CastExpression(
            empty_data,
            in_expr,
        )

    if (
        operator_class_name == "SqlPostfixOperator"
        and len(java_call.getOperands()) == 1
    ):
        operands = java_call.getOperands()
        input = java_expr_to_python_expr(ctx, operands[0], input_plan)
        kind = op.getKind()
        SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind

        if kind.equals(SqlKind.IS_NOT_NULL):
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return UnaryOpExpression(bool_empty_data, input, "notnull")

        if kind.equals(SqlKind.IS_NULL):
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return UnaryOpExpression(bool_empty_data, input, "isnull")

        if kind.equals(SqlKind.IS_TRUE):
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return UnaryOpExpression(bool_empty_data, input, "istrue")

    if operator_class_name == "SqlCaseOperator":
        operands = java_call.getOperands()
        kind = op.getKind()
        SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind
        assert kind.equals(SqlKind.CASE), (
            "Expected CASE operator, got " + kind.toString()
        )

        return java_case_to_python_case(ctx, operands, input_plan)

    if operator_class_name == "SqlPrefixOperator" and len(java_call.getOperands()) == 1:
        operands = java_call.getOperands()
        input = java_expr_to_python_expr(ctx, operands[0], input_plan)
        kind = op.getKind()
        SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind

        if kind.equals(SqlKind.MINUS_PREFIX):
            out_empty = -input.empty_data.iloc[:, 0]
            return UnaryOpExpression(out_empty, input, "__neg__")

        if kind.equals(SqlKind.NOT):
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return UnaryOpExpression(bool_empty_data, input, "__invert__")

    if (
        operator_class_name == "SqlExtractFunction"
        and len(java_call.getOperands()) == 2
    ):
        # EXTRACT(FLAG(MONTH), date) → month(date)
        unit_str = str(java_call.getOperands()[0].toString()).upper()
        input = java_expr_to_python_expr(ctx, java_call.getOperands()[1], input_plan)
        # Strip FLAG from unit if it's in the form FLAG(unit)
        if "(" in unit_str:
            unit_str = unit_str.split("(")[1].rstrip(")")
        arrow_func = _DATE_PART_ARROW_FUNCS.get(unit_str)
        if arrow_func is None:
            raise NotImplementedError(f"Unsupported EXTRACT unit: {unit_str}")
        empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
        raw_expr = ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())
        if unit_str in ("DAYOFWEEK", "DOW"):
            # PyArrow default: 0=Monday, 6=Sunday.
            # Bodo/Snowflake DAYOFWEEK: 1=Monday, 0=Sunday → (dow+1)%7
            one = ConstantExpression(empty_data, input_plan, 1)
            seven = ConstantExpression(empty_data, input_plan, 7)
            raw_expr = ArithOpExpression(empty_data, raw_expr, one, "__add__")
            raw_expr = ArithOpExpression(empty_data, raw_expr, seven, "__mod__")
        # For WEEKDAY, PyArrow default (0=Monday..6=Sunday) exactly matches
        # Spark/Snowflake WEEKDAY, so no transformation is needed.
        return raw_expr
    if (
        operator_class_name == "SqlDatePartFunction"
        and len(java_call.getOperands()) == 1
    ):
        operands = java_call.getOperands()
        input = java_expr_to_python_expr(ctx, operands[0], input_plan)
        func_name = op.getName().upper()

        # Map Calcite function names to Arrow compute function names
        arrow_func = _DATE_PART_ARROW_FUNCS.get(func_name, func_name.lower())

        if func_name in (
            "YEAR",
            "MONTH",
            "DAY",
            "DAYOFMONTH",
            "DAYOFYEAR",
            "WEEK",
            "WEEKOFYEAR",
            "WEEKISOHOUR",
            "HOUR",
            "MINUTE",
            "SECOND",
            "QUARTER",
            "MICROSECOND",
            "NANOSECOND",
        ):
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            return ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())

        if func_name in ("WEEK", "WEEKOFYEAR", "WEEKISO", "DAYOFYEAR"):
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            return ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())

        if func_name in ("DAYOFWEEK", "DOW"):
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            dow_expr = ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())
            # PyArrow default: 0=Monday, 6=Sunday.
            # Bodo/Snowflake DAYOFWEEK: 1=Monday, 0=Sunday → (dow+1)%7
            one_const = ConstantExpression(empty_data, input_plan, 1)
            seven_const = ConstantExpression(empty_data, input_plan, 7)
            plus_one = ArithOpExpression(empty_data, dow_expr, one_const, "__add__")
            return ArithOpExpression(empty_data, plus_one, seven_const, "__mod__")

        if func_name == "WEEKDAY":
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            # PyArrow default (0=Monday..6=Sunday) exactly matches
            # Spark/Snowflake WEEKDAY, so no transformation is needed.
            return ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())

    if operator_class_name == "SqlCoalesceFunction":
        operands = java_call.getOperands()
        op_exprs = [java_expr_to_python_expr(ctx, o, input_plan) for o in operands]
        # Unify data types to match output type of coalesce (e.g. int8 + int32 -> int32)
        out_col_name = op_exprs[0].empty_data.columns[0]
        in_schemas = [
            pa.Schema.from_pandas(e.empty_data.set_axis([out_col_name], axis=1))
            for e in op_exprs
        ]
        # If some but not all inputs are timestamps, promote all to timestamps to
        # avoid errors in C++ backend
        if any(pa.types.is_timestamp(s.field(0).type) for s in in_schemas) and not all(
            pa.types.is_timestamp(s.field(0).type) for s in in_schemas
        ):
            t = next(
                s.field(0).type
                for s in in_schemas
                if pa.types.is_timestamp(s.field(0).type)
            )
            out_schema = pa.schema([pa.field(out_col_name, t)])
        else:
            out_schema = pa.unify_schemas(
                in_schemas,
                promote_options="permissive",
            )
        empty_data = arrow_to_empty_df(out_schema)
        return ArrowScalarFuncExpression(empty_data, op_exprs, "coalesce", ())

    if operator_class_name == "SqlCurrentDateFunction":
        # Matching BodoSQL JIT backend which uses UTC by default
        # https://github.com/bodo-ai/Bodo/blob/c151771c58a61753daba450901eb294a76b8ff58/BodoSQL/calcite_sql/bodosql-calcite-application/src/main/java/com/bodosql/calcite/application/BodoSQLCodeGen/DatetimeFnCodeGen.java#L260
        # https://github.com/bodo-ai/Bodo/blob/c151771c58a61753daba450901eb294a76b8ff58/bodo/hiframes/datetime_date_ext.py#L1252
        tz_info = zoneinfo.ZoneInfo("UTC")
        curr_date = datetime.now(tz_info).date()
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.date32()))
        # NOTE: this is assuming that plans are not cached so has to be changed if we
        # add plan caching
        return ConstantExpression(dummy_empty_data, input_plan, curr_date)

    if operator_class_name == "SqlAbstractTimeFunction":
        func_name = op.getName().upper()
        if func_name in ("LOCALTIME", "CURRENT_TIME"):
            curr_ts = pd.Timestamp.now()
            curr_time = pc.cast(curr_ts, pa.time64("ns"))
            dummy_empty_data = pd.Series(
                [curr_time], dtype=pd.ArrowDtype(pa.time64("ns"))
            )
            return ConstantExpression(dummy_empty_data, input_plan, curr_time)

    if operator_class_name == "SqlBasicFunction":
        # Map Calcite basic functions to Bodo expressions
        operands = java_call.getOperands()
        op_exprs = [java_expr_to_python_expr(ctx, o, input_plan) for o in operands]
        # function name as string (e.g., "POWER", "SQRT")
        func_name = op.getName().upper()

        if func_name in ("UTC_TIMESTAMP", "UTC_DATE"):
            curr_ts = pd.Timestamp.now()
            if func_name == "UTC_DATE":
                curr_ts = curr_ts.normalize()
            dummy_empty_data = pd.Series(
                [curr_ts], dtype=pd.ArrowDtype(pa.timestamp("ns"))
            )
            return ConstantExpression(dummy_empty_data, input_plan, curr_ts)

        if func_name in (
            "CURRENT_TIMESTAMP",
            "GETDATE",
            "LOCALTIMESTAMP",
            "SYSTIMESTAMP",
            "NOW",
        ):
            tz = ctx.default_tz if ctx.default_tz is not None else "UTC"
            curr_ts = pd.Timestamp.now(tz=tz)
            dummy_empty_data = pd.Series(
                [curr_ts], dtype=pd.ArrowDtype(pa.timestamp("ns", tz=tz))
            )
            return ConstantExpression(dummy_empty_data, input_plan, curr_ts)

        # COMBINE_INTERVALS combines multiple interval literals into a single
        # interval constant. Accumulate months (from DateOffset) and nanoseconds
        # (from Timedelta) separately since pd.DateOffset does not support
        # arithmetic with other DateOffsets or Timedeltas.
        if func_name == "COMBINE_INTERVALS":
            total_months = 0
            total_nanos = 0
            for expr in op_exprs:
                assert isinstance(expr, ConstantExpression), (
                    "COMBINE_INTERVALS requires constant interval arguments in C++ backend"
                )
                val = expr.value
                if (
                    isinstance(val, tuple)
                    and len(val) == 4
                    and val[0] == "MonthDayNanoInterval"
                ):
                    total_months += val[1]
                    total_nanos += val[3]
                elif isinstance(val, pd.DateOffset):
                    total_months += val.months
                elif isinstance(val, pd.Timedelta):
                    total_nanos += val.value
                else:
                    raise ValueError(
                        f"Unexpected interval type in COMBINE_INTERVALS: {type(val)}"
                    )
            if total_nanos % 1000 != 0:
                raise ValueError(
                    "Sub-microsecond intervals not supported in C++ backend"
                )
            if total_months != 0:
                combined_val = ("MonthDayNanoInterval", total_months, 0, total_nanos)
            else:
                combined_val = pd.Timedelta(nanoseconds=total_nanos)
            dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.duration("ns")))
            return ConstantExpression(dummy_empty_data, input_plan, combined_val)

        # Binary power: POWER(x, y) -> use __pow__ via ArithOpExpression
        if func_name == "POWER" and len(op_exprs) == 2:
            left = op_exprs[0]
            right = op_exprs[1]
            out_empty = left.empty_data.iloc[:, 0] ** right.empty_data.iloc[:, 0]
            return ArithOpExpression(out_empty, left, right, "__pow__")

        # SQRT(x) -> unary sqrt
        if func_name == "SQRT" and len(op_exprs) == 1:
            inp = op_exprs[0]
            out_empty = inp.empty_data.iloc[:, 0] ** 0.5
            return UnaryOpExpression(out_empty, inp, "sqrt")

        # ABS(x)
        if func_name == "ABS" and len(op_exprs) == 1:
            inp = op_exprs[0]
            out_empty = inp.empty_data.iloc[:, 0].abs()
            return UnaryOpExpression(out_empty, inp, "abs")

        # CEIL(x) / CEILING(x)
        if func_name in ("CEIL", "CEILING") and len(op_exprs) == 1:
            inp = op_exprs[0]
            out_empty = inp.empty_data
            return UnaryOpExpression(out_empty, inp, "ceil")

        # FLOOR(x)
        if func_name == "FLOOR" and len(op_exprs) == 1:
            inp = op_exprs[0]
            out_empty = inp.empty_data
            return UnaryOpExpression(out_empty, inp, "floor")

        # EXP(x)
        if func_name == "EXP" and len(op_exprs) == 1:
            inp = op_exprs[0]
            out_empty = inp.empty_data
            out_empty = out_empty.astype("float64")
            return UnaryOpExpression(out_empty, inp, "exp")

        # LN(x) or LOG(x) -> natural log
        if func_name in ("LN", "LOG") and len(op_exprs) == 1:
            inp = op_exprs[0]
            out_empty = inp.empty_data
            out_empty = out_empty.astype("float64")
            return UnaryOpExpression(out_empty, inp, "log")

        # ROUND(x, d) or ROUND(x) -> map to a unary/binary op if supported
        if func_name == "ROUND" and len(op_exprs) in (1, 2):
            inp = op_exprs[0]
            out_empty = inp.empty_data
            return UnaryOpExpression(out_empty, inp, "round")

        if func_name == "MOD" and len(op_exprs) == 2:
            inp = op_exprs[0]
            modulus_expr = op_exprs[1]
            ensure_type_of_expr(inp, "MOD inp", int)
            ensure_type_of_expr(modulus_expr, "modulus_expr", int)

            return ArithOpExpression(inp.empty_data, inp, modulus_expr, "__mod__")

        if func_name == "IFF" and len(op_exprs) == 3:
            # IFF is equivalent to CASE with single WHEN
            return java_case_to_python_case(ctx, operands, input_plan)

        if func_name == "NULLIF" and len(op_exprs) == 2:
            return ArrowScalarFuncExpression(
                op_exprs[0].empty_data, op_exprs, "nullif", ()
            )

        if func_name == "CHAR_LENGTH" and len(op_exprs) == 1:
            src = op_exprs[0]
            ensure_type_of_expr(src, "src", str)
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            return ArrowScalarFuncExpression(
                int_empty_data,
                [src],
                "utf8_length",
                (),
            )

        if func_name == "LOWER" and len(op_exprs) == 1:
            return ArrowScalarFuncExpression(
                op_exprs[0].empty_data, op_exprs, "utf8_lower", ()
            )

        if func_name == "UPPER" and len(op_exprs) == 1:
            return ArrowScalarFuncExpression(
                op_exprs[0].empty_data, op_exprs, "utf8_upper", ()
            )

        if func_name in ("LPAD", "RPAD") and len(op_exprs) in (2, 3):
            src = op_exprs[0]
            ensure_type_of_expr(src, "src", str)

            length = op_exprs[1]
            ensure_arg_is_const_expr_of_type(length, "length", int)

            arrow_func_args = (length.value,)

            if len(op_exprs) == 3:
                pattern = op_exprs[2]
                ensure_arg_is_const_expr_of_type(pattern, "pattern", str)
                arrow_func_args += (pattern.value,)

            return ArrowScalarFuncExpression(
                src.empty_data,
                [src],
                f"utf8_{func_name.lower()}",
                arrow_func_args,
            )

        if func_name == "REPLACE" and len(op_exprs) in (2, 3):
            src = op_exprs[0]
            search_expr = op_exprs[1]

            ensure_type_of_expr(src, "src", str)
            ensure_arg_is_const_expr_of_type(search_expr, "search_expr", str)

            if len(op_exprs) == 3:
                replacement_expr = op_exprs[2]
                ensure_arg_is_const_expr_of_type(
                    replacement_expr, "replacement_expr", str
                )
                replacement_val = replacement_expr.value
            else:
                replacement_val = ""

            return ArrowScalarFuncExpression(
                src.empty_data,
                [src],
                "replace_substring",
                (search_expr.value, replacement_val),
            )

        if func_name == "REGEXP_SUBSTR" and len(op_exprs) in (2, 3, 4, 5, 6):
            src = op_exprs[0]
            regexp = op_exprs[1]

            ensure_type_of_expr(src, "src", str)
            ensure_arg_is_const_expr_of_type(regexp, "regexp", str)

            if len(op_exprs) >= 3:
                start_expr = op_exprs[2]
                ensure_arg_is_const_expr_of_type(start_expr, "start_expr", int)

                if start_expr.value > 0:
                    start = start_expr.value - 1
                else:
                    start = start_expr.value
            else:
                start = 0

            if len(op_exprs) >= 4:
                # Need to search for the substring that is the op_exprs[3]-th occurrence / regex match
                occurrence_expr = op_exprs[3]
                ensure_arg_is_const_expr_of_type(
                    occurrence_expr, "occurrence_expr", int
                )
                occurrence_num = occurrence_expr.value
                if occurrence_num < 1:
                    raise ValueError(
                        f"{func_name} occurences argument must be 1 or greater"
                    )
            else:
                occurrence_num = 1

            if len(op_exprs) >= 5:
                regex_params_expr = op_exprs[4]
                ensure_arg_is_const_expr_of_type(
                    regex_params_expr, "regex_params_expr", str
                )
                if "c" in regex_params_expr.value and "i" in regex_params_expr.value:
                    # Both case sensitive and case insensitive params provided; find which appears latest in the string
                    latest_index = max(
                        regex_params_expr.value.rfind(char) for char in ("c", "i")
                    )
                    latest_char = regex_params_expr.value[latest_index]
                    # Remove occurrences of the other parameter to make the identification easier on the C++ side
                    regex_params = regex_params_expr.value.replace(
                        "c" if latest_char == "i" else "i", ""
                    )
                else:
                    if (
                        "c" not in regex_params_expr.value
                        and "i" not in regex_params_expr.value
                    ):
                        regex_params = regex_params_expr.value + "c"
                    else:
                        regex_params = regex_params_expr.value
                for character in regex_params:
                    if character not in ("c", "i", "m", "e", "s"):
                        raise ValueError(
                            f"{func_name} regex parameter {character} does not exist"
                        )
                    if character in ("i", "m", "s"):
                        raise ValueError(
                            f"{func_name} regex parameter {character} is not yet supported in the C++ backend"
                        )
            else:
                regex_params = "c"

            if len(op_exprs) == 6:
                group_num_expr = op_exprs[5]
                ensure_arg_is_const_expr_of_type(group_num_expr, "group_num_expr", int)
                if group_num_expr.value < 0:
                    raise ValueError(
                        f"Negative value for group_num argument of {func_name} is not permitted"
                    )
                group_num = group_num_expr.value
                if group_num > 0:
                    group_num -= 1  # Convert from 1-based to 0-based
                regex_params = (
                    regex_params + "e"
                )  # 'e' is implied if group_num is passed
            else:
                group_num = 0

            # Chop off the start so that searching begins after the provided position
            without_start_expr = ArrowScalarFuncExpression(
                src.empty_data,
                [src],
                "utf8_slice_codeunits",
                (start, None, 1),
            )

            # Remove earlier occurrences so that extract_regex can find the correct occurrence/substring matching the regexp
            # TODO: How should this behave for overlapping occurrences?
            if occurrence_num > 1:
                occurences_replaced_expr = ArrowScalarFuncExpression(
                    src.empty_data,
                    [without_start_expr],
                    "replace_substring_regex",
                    (
                        regexp.value,
                        "",
                        occurrence_num - 1,
                    ),
                )
            else:
                occurences_replaced_expr = without_start_expr

            return ArrowScalarFuncExpression(
                src.empty_data,
                [occurences_replaced_expr],
                "regexp_substr",  # Made up function, will redirect to extract_regex with the right group extracted
                (regexp.value, regex_params, group_num),
            )

        # If we didn't match a supported basic function, fall through to NotImplemented
        raise NotImplementedError(
            f"SqlBasicFunction {func_name} not supported yet: " + java_call.toString()
        )

    if operator_class_name == "SqlNullPolicyFunction":
        operands = java_call.getOperands()
        op_exprs = [java_expr_to_python_expr(ctx, o, input_plan) for o in operands]
        func_name = op.getName().upper()

        if (
            func_name in ("BITAND", "BITOR", "BITXOR", "BITSHIFTLEFT", "BITSHIFTRIGHT")
            and len(op_exprs) == 2
        ):
            left_expr = op_exprs[0]
            right_expr = op_exprs[1]

            ensure_type_of_expr(left_expr, "left_expr", int)
            ensure_arg_is_const_expr_of_type(right_expr, "right_expr", int)

            empty_data = left_expr.empty_data

            if func_name == "BITAND":
                arrow_equivalent_func = "bit_wise_and"
            elif func_name == "BITOR":
                arrow_equivalent_func = "bit_wise_or"
            elif func_name == "BITXOR":
                arrow_equivalent_func = "bit_wise_xor"
            elif func_name == "BITSHIFTLEFT":
                arrow_equivalent_func = "shift_left"
                # Left shift can overflow, so promote to INT64
                empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            elif func_name == "BITSHIFTRIGHT":
                arrow_equivalent_func = "shift_right"

            return ArrowScalarFuncExpression(
                empty_data, [left_expr], arrow_equivalent_func, (right_expr.value,)
            )
        elif func_name == "BITNOT" and len(op_exprs) == 1:
            src = op_exprs[0]
            ensure_type_of_expr(src, "src", int)
            return ArrowScalarFuncExpression(src.empty_data, [src], "bit_wise_not", ())
        elif func_name == "GETBIT" and len(op_exprs) == 2:
            src = op_exprs[0]
            bit_num = op_exprs[1]

            ensure_type_of_expr(src, "src", int)
            ensure_arg_is_const_expr_of_type(bit_num, "bit_num", int)
            if bit_num.value < 0:
                raise ValueError("GETBIT position cannot be negative")

            # Do a bitwise AND on `src` and a bitmask that is only set on the requested bit position.
            # If the result is nonzero, the requested bit is 1, else it is 0.

            int64_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            src_with_mask = ArrowScalarFuncExpression(
                int64_empty_data, [src], "bit_wise_and", (1 << bit_num.value,)
            )

            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            is_bit_set = ArrowScalarFuncExpression(
                bool_empty_data, [src_with_mask], "not_equal", (0,)
            )

            return ArrowScalarFuncExpression(
                int64_empty_data, [is_bit_set], "if_else", (1, 0)
            )
        elif func_name == "LEFT" and len(op_exprs) == 2:
            # Implement LEFT as substr(0,...)
            src = op_exprs[0]
            len_expr = op_exprs[1]

            ensure_type_of_expr(src, "src", str)
            ensure_arg_is_const_expr_of_type(len_expr, "len_expr", int)

            out_empty = src.empty_data.iloc[:, 0]
            return ArrowScalarFuncExpression(
                out_empty, [src], "utf8_slice_codeunits", (0, len_expr.value, 1)
            )
        elif func_name == "RIGHT" and len(op_exprs) == 2:
            # Implement RIGHT as substr(-len,...)
            src = op_exprs[0]
            len_expr = op_exprs[1]

            ensure_type_of_expr(src, "src", str)
            ensure_arg_is_const_expr_of_type(len_expr, "len_expr", int)

            out_empty = src.empty_data.iloc[:, 0]
            return ArrowScalarFuncExpression(
                out_empty, [src], "utf8_slice_codeunits", (-len_expr.value, None, 1)
            )
        elif func_name == "STARTSWITH" and len(op_exprs) == 2:
            src = op_exprs[0]
            match_expr = op_exprs[1]

            ensure_type_of_expr(src, "src", str)
            ensure_arg_is_const_expr_of_type(match_expr, "match_expr", str)

            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return ArrowScalarFuncExpression(
                bool_empty_data,
                [src],
                "starts_with",
                (match_expr.value,),
            )
        elif func_name == "ENDSWITH" and len(op_exprs) == 2:
            src = op_exprs[0]
            match_expr = op_exprs[1]

            ensure_type_of_expr(src, "src", str)
            ensure_arg_is_const_expr_of_type(match_expr, "match_expr", str)

            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return ArrowScalarFuncExpression(
                bool_empty_data,
                [src],
                "ends_with",
                (match_expr.value,),
            )
        elif func_name == "CONTAINS" and len(op_exprs) == 2:
            src = op_exprs[0]
            match_expr = op_exprs[1]

            ensure_type_of_expr(src, "src", str)
            ensure_arg_is_const_expr_of_type(match_expr, "match_expr", str)

            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return ArrowScalarFuncExpression(
                bool_empty_data,
                [src],
                "match_substring",
                (match_expr.value,),
            )
        elif func_name == "LENGTH" and len(op_exprs) == 1:
            src = op_exprs[0]
            ensure_type_of_expr(src, "src", str)
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            return ArrowScalarFuncExpression(
                int_empty_data,
                [src],
                "utf8_length",
                (),
            )
        elif (
            func_name == "INSTR"
            and len(op_exprs) in (2, 3, 4)
            or func_name == "CHARINDEX"
            and len(op_exprs) in (2, 3)
        ):
            # TODO: Investigate what the proper outputs should be in various exceptional conditions

            src = op_exprs[0]
            match_expr = op_exprs[1]

            ensure_type_of_expr(src, "src", str)
            ensure_arg_is_const_expr_of_type(match_expr, "match_expr", str)
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            if len(match_expr.value) == 0:
                return ConstantExpression(int_empty_data, input_plan, 0)

            if len(op_exprs) > 2:
                start_expr = op_exprs[2]
                ensure_arg_is_const_expr_of_type(start_expr, "start_expr", int)

                if start_expr.value > 0:
                    start = start_expr.value - 1
                else:
                    start = start_expr.value
            else:
                start = 0

            if len(op_exprs) == 4:
                # Need to search for the position of the op_exprs[3]-th occurrence of the substring
                occurrence_expr = op_exprs[3]
                ensure_arg_is_const_expr_of_type(
                    occurrence_expr, "occurrence_expr", int
                )
                occurrence_num = occurrence_expr.value
                if occurrence_num < 1:
                    raise ValueError(
                        f"{func_name} occurences argument must be 1 or greater"
                    )
            else:
                occurrence_num = 1

            without_start_expr = ArrowScalarFuncExpression(
                src.empty_data,
                [src],
                "utf8_slice_codeunits",
                (start, None, 1),
            )

            # Since Arrow only has a compute function to search for the first occurrence, we must workaround this.
            # Our solution is to slightly rename earlier occurrences so that find_substring can find the first of those remaining
            # The index should be the same since we do not delete any characters

            # Replace occurrences of the substring before the one we want the position of
            if occurrence_num > 1:
                first_char_replacement = "_" if match_expr.value[0] != "_" else "-"
                occurences_replaced_expr = ArrowScalarFuncExpression(
                    src.empty_data,
                    [without_start_expr],
                    "replace_substring",
                    (
                        match_expr.value,
                        first_char_replacement + match_expr.value[1:],
                        occurrence_num - 1,
                    ),
                )
            else:
                occurences_replaced_expr = without_start_expr

            # Find the first occurrence of the substring in the sliced string (without prior occurrences)
            substring_pos_expr = ArrowScalarFuncExpression(
                int_empty_data,
                [occurences_replaced_expr],
                "find_substring",
                (match_expr.value,),
            )

            start_const_expr = ConstantExpression(int_empty_data, input_plan, start)
            occurrence_pos_expr = ArithOpExpression(
                int_empty_data, substring_pos_expr, start_const_expr, "__add__"
            )

            # Add 1 to find_substring expression since Arrow's find_substring is 0-indexed instead of 1-based like INSTR/CHARINDEX
            one = ConstantExpression(int_empty_data, input_plan, 1)
            return ArithOpExpression(
                int_empty_data, occurrence_pos_expr, one, "__add__"
            )

        elif func_name == "INITCAP" and len(op_exprs) in (1, 2):
            raise ValueError("INITCAP currently disabled on C++ backend")

            src = op_exprs[0]
            if len(op_exprs) == 2:
                delim_expr = op_exprs[1]
                ensure_arg_is_const_expr_of_type(delim_expr, "delim_expr", str)
                raise ValueError("Delimiter argument to INITCAP not yet supported")
            # Note that Arrow's utf8_title considers numbers to be delimiters, which may differ from some implementations of INITCAP
            return ArrowScalarFuncExpression(
                src.empty_data,
                [src],
                "utf8_title",
                (),
            )
        elif func_name == "CONCAT" and len(op_exprs) > 0:
            src = op_exprs[0]
            ensure_type_of_expr(src, "src", str)

            if len(op_exprs) == 1:
                # Nothing to concatenate, just return the input string
                return src

            separator = bodo.pandas.plan.ConstantExpression(
                src.empty_data,
                src.source,
                "",  # empty separator to concat without anything in between
            )

            input_exprs = [src]

            if len(op_exprs) > 1:
                for other_str_src in op_exprs[1:]:
                    ensure_type_of_expr(other_str_src, "other_str_src", str)
                    input_exprs.append(other_str_src)

            input_exprs.append(separator)

            return ArrowScalarFuncExpression(
                src.empty_data,
                input_exprs,
                "binary_join_element_wise",
                (),
            )
        elif func_name == "CONCAT_WS" and len(op_exprs) > 1:
            separator = op_exprs[0]
            ensure_type_of_expr(separator, "separator", str)

            if len(op_exprs) == 2:
                # Nothing to concatenate, just return the input string
                return op_exprs[1]

            input_exprs = []
            for str_src in op_exprs[1:]:
                ensure_type_of_expr(str_src, "str_src", str)
                input_exprs.append(str_src)
            input_exprs.append(separator)

            return ArrowScalarFuncExpression(
                input_exprs[0].empty_data,
                input_exprs,
                "binary_join_element_wise",
                (
                    "skip",
                ),  # Ensure null string arguments are treated as empty / skipped
            )
        elif func_name == "REPEAT" and len(op_exprs) == 2:
            src = op_exprs[0]
            num_repeats_expr = op_exprs[1]

            ensure_type_of_expr(src, "src", str)
            ensure_arg_is_const_expr_of_type(num_repeats_expr, "num_repeats_expr", int)

            return ArrowScalarFuncExpression(
                src.empty_data, [src], "binary_repeat", (num_repeats_expr.value,)
            )
        elif func_name == "SPACE" and len(op_exprs) == 1:
            num_repeats_expr = op_exprs[0]
            ensure_arg_is_const_expr_of_type(num_repeats_expr, "num_repeats_expr", int)

            str_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.string()))

            space_expr = bodo.pandas.plan.ConstantExpression(
                str_empty_data, input_plan, " "
            )

            return ArrowScalarFuncExpression(
                str_empty_data, [space_expr], "binary_repeat", (num_repeats_expr.value,)
            )
        elif func_name == "REVERSE" and len(op_exprs) == 1:
            src = op_exprs[0]
            ensure_type_of_expr(src, "src", str)

            return ArrowScalarFuncExpression(src.empty_data, [src], "utf8_reverse", ())
        elif func_name == "RTRIMMED_LENGTH" and len(op_exprs) == 1:
            src = op_exprs[0]
            ensure_type_of_expr(src, "src", str)
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            # Use utf8_rtrim instead of utf8_rtrim_whitespace so that only regular space characters are removed
            rtrimmed_expr = ArrowScalarFuncExpression(
                src.empty_data, [src], "utf8_rtrim", (" ",)
            )
            return ArrowScalarFuncExpression(
                int_empty_data, [rtrimmed_expr], "utf8_length", ()
            )
        elif func_name == "INSERT" and len(op_exprs) == 4:
            src = op_exprs[0]
            start_expr = op_exprs[1]
            len_expr = op_exprs[2]
            inserted_str_expr = op_exprs[3]

            ensure_type_of_expr(src, "src", str)
            ensure_arg_is_const_expr_of_type(start_expr, "start_expr", int)
            ensure_arg_is_const_expr_of_type(len_expr, "len_expr", int)
            ensure_arg_is_const_expr_of_type(
                inserted_str_expr, "inserted_str_expr", str
            )

            return ArrowScalarFuncExpression(
                src.empty_data,
                [src],
                "utf8_replace_slice",
                (
                    start_expr.value - 1,
                    start_expr.value - 1 + len_expr.value,
                    inserted_str_expr.value,
                ),
            )

    if operator_class_name == "SqlSubstringFunction":
        operands = java_call.getOperands()
        op_exprs = [java_expr_to_python_expr(ctx, o, input_plan) for o in operands]
        func_name = op.getName().upper()

        if func_name == "SUBSTRING" and len(op_exprs) in (2, 3):
            # See:
            # https://github.com/bodo-ai/Bodo/blob/88f6a82ee1ffedbdf7370a37b7bee7ad93982413/BodoSQL/bodosql/kernels/string_array_kernels.py#L1993
            # https://docs.bodo.ai/latest/api_docs/sql/functions/string/substring/#substring
            src = op_exprs[0]
            start_expr = op_exprs[1]
            ensure_arg_is_const_expr_of_type(start_expr, "start_expr", int)

            start = start_expr.value
            if (
                start > 0
            ):  # start_expr.value = 0 is treated the same as start_expr.value = 1
                start -= 1  # SQL substring is 1-indexed but Arrow is 0-indexed
            # Arrow's utf8_slice_codeunits will handle the wraparound for negative start index

            if len(op_exprs) == 3:
                len_expr = op_exprs[2]
                ensure_arg_is_const_expr_of_type(len_expr, "len_expr", int)
                if len_expr.value < 0:
                    raise ValueError(
                        "negative length not allowed in SUBSTRING in C++ backend"
                    )
                stop = start + len_expr.value
                # Deal with negative start index and length beyond the end of the string
                if start < 0 and stop >= 0:
                    stop = None
            else:
                stop = None

            out_empty = src.empty_data.iloc[:, 0]
            return ArrowScalarFuncExpression(
                out_empty,
                [src],
                "utf8_slice_codeunits",
                (start, stop, 1),
            )

    if operator_class_name == "SqlLikeOperator":
        operands = java_call.getOperands()
        op_exprs = [java_expr_to_python_expr(ctx, o, input_plan) for o in operands]
        func_name = op.getName().upper()

        if func_name in ("LIKE", "ILIKE") and len(op_exprs) in (2, 3):
            left = op_exprs[0]
            like_expr = op_exprs[1]
            if len(op_exprs) == 3:
                ensure_arg_is_const_expr_of_type(op_exprs[2], "escape_expr", str)
                escape_val = op_exprs[2].value
            else:
                escape_val = ""
            ensure_arg_is_const_expr_of_type(like_expr, "like_expr", str)

            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            converted_like, needs_regex, start_match, end_match, match_anything = (
                bodo.ir.filter.convert_sql_pattern_to_python_compile_time(
                    like_expr.value, escape_val, False
                )
            )
            if needs_regex:
                if match_anything:
                    raise NotImplementedError(
                        "LIKE conversion supports nothing else if regex is required."
                    )
                return ArrowScalarFuncExpression(
                    bool_empty_data,
                    [left],
                    "match_substring_regex",
                    (converted_like, func_name == "ILIKE"),
                )
            elif start_match:
                if end_match or match_anything:
                    raise NotImplementedError(
                        "LIKE conversion supports nothing else if start_match is required."
                    )
                return ArrowScalarFuncExpression(
                    bool_empty_data,
                    [left],
                    "starts_with",
                    (converted_like, func_name == "ILIKE"),
                )
            elif end_match:
                if match_anything:
                    raise NotImplementedError(
                        "LIKE conversion supports nothing else if end_match is required."
                    )
                return ArrowScalarFuncExpression(
                    bool_empty_data,
                    [left],
                    "ends_with",
                    (converted_like, func_name == "ILIKE"),
                )
            elif match_anything:
                raise NotImplementedError(
                    "LIKE conversion does not currently support match anything."
                )
            else:
                return ArrowScalarFuncExpression(
                    bool_empty_data,
                    [left],
                    "match_substring",
                    (converted_like, func_name == "ILIKE"),
                )

    if operator_class_name == "SqlSearchOperator":
        operands = java_call.getOperands()
        op_exprs = [java_expr_to_python_expr(ctx, o, input_plan) for o in operands]
        func_name = op.getName().upper()

        if func_name == "SEARCH" and len(op_exprs) == 2:
            src = op_exprs[0]
            # search_expr is a ConstantExpression with org.apache.calcite.util.Sarg value
            search_expr = op_exprs[1]
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            # sarg is an org.apache.calcite.util.Sarg
            sarg = search_expr.value
            assert sarg.getClass().getSimpleName() == "Sarg"
            if (
                sarg.getClass().getDeclaredField("nullAs").get(sarg).toString()
                != "UNKNOWN"
            ):
                raise NotImplementedError(
                    "SEARCH operator with nullAs not UNKNOWN not supported in C++ backend yet"
                )
            # sarg_rangeSet is a com.google.common.collect.ImmutableRangeSet
            sarg_rangeSet = sarg.getClass().getDeclaredField("rangeSet").get(sarg)
            assert sarg_rangeSet.getClass().getSimpleName() == "ImmutableRangeSet"
            # ranges_collection is a com.google.common.collect.RegularImmutableSortedSet
            ranges_collection = sarg_rangeSet.asRanges()
            assert (
                ranges_collection.getClass().getSimpleName()
                == "RegularImmutableSortedSet"
            )
            search_options = []
            it = ranges_collection.iterator()
            while it.hasNext():
                # r is a com.google.common.collect.Range
                r = it.next()
                assert r.getClass().getSimpleName() == "Range"
                search_options.append(r)

            def range_type_to_python(x):
                if isinstance(x, py4j.java_gateway.JavaObject):
                    return x.getValue()
                elif isinstance(x, decimal.Decimal):
                    return float(x)
                else:
                    return x

            def process_one_search_option(so):
                """Generate an expression to check if src satisfies this
                current possibility from the range set."""
                # Get and convert the lower range endpoint if it has one else None.
                lower_lit = (
                    range_type_to_python(so.lowerEndpoint())
                    if so.hasLowerBound()
                    else None
                )
                # Get and convert the upper range endpoint if it has one else None.
                upper_lit = (
                    range_type_to_python(so.upperEndpoint())
                    if so.hasUpperBound()
                    else None
                )
                lower_inclusive = so.lowerBoundType().toString() == "CLOSED"
                upper_inclusive = so.upperBoundType().toString() == "CLOSED"
                # There are 9 possibilities here.  NA=Not applicable
                # HasLower HasUpper LowerInclusive UpperInclusive
                # F        F        NA             NA
                # T        F        T              NA
                # T        F        F              NA
                # F        T        NA             T
                # F        T        NA             F
                # T        T        T              T
                # T        T        T              F
                # T        T        F              T
                # T        T        F              F

                # Also, the case of having lower and upper bound and both being
                # inclusive can be divided up into two cases, where lower and
                # upper are equal and where they are not equal.
                if (
                    lower_lit is not None
                    and upper_lit is not None
                    and lower_inclusive
                    and upper_inclusive
                    and lower_lit == upper_lit
                ):
                    # The exception case where we have lower and upper bounds and they are both
                    # inclusive and the same we can simplify as equality check.
                    const_empty_data = arrow_to_empty_df(
                        pa.schema([pa.field("equal", pa.scalar(lower_lit).type)])
                    )

                    return ComparisonOpExpression(
                        bool_empty_data,
                        src,
                        ConstantExpression(const_empty_data, input_plan, lower_lit),
                        operator.eq,
                    )

                raise NotImplementedError(
                    f"SEARCH operator case of hasLower {lower_lit is not None}, hasUpper {upper_lit is not None}, lowerInclusive {lower_inclusive}, upperInclusive {upper_inclusive} not supported yet."
                )

            out_expr = process_one_search_option(search_options[0])
            # The definition of search is that the value is one of the
            # possibilities in the range set.  so, "or" in the other
            # possibilities below.
            for so in search_options[1:]:
                out_expr = ConjunctionOpExpression(
                    bool_empty_data, out_expr, process_one_search_option(so), "__or__"
                )
            return out_expr

        raise NotImplementedError(
            f"Function name {func_name} not supported for SEARCH operator yet: "
            + java_call.toString()
        )

    raise NotImplementedError(
        f"Call operator {operator_class_name} not supported yet: "
        + java_call.toString()
    )


def ensure_arg_is_const_expr_of_type(expr, expr_name, dtype):
    if not isinstance(expr, bodo.pandas.plan.ConstantExpression):
        raise ValueError(
            f"{expr_name} should be ConstantExpression but instead was {type(expr)}"
        )
    if not isinstance(expr.value, dtype):
        raise ValueError(
            f"{expr_name}.value should be {str(dtype)} but instead was {type(expr.value)}"
        )


def ensure_type_of_expr(expr, expr_name, dtype):
    def compare_types(obj_type, expected_type):
        if expected_type is int:
            return pd.api.types.is_integer_dtype(obj_type)
        if expected_type is float:
            return pd.api.types.is_float_dtype(obj_type)
        if expected_type is str:
            return pd.api.types.is_string_dtype(obj_type)
        if expected_type is bool:
            return pd.api.types.is_bool_dtype(obj_type)
        if isinstance(expected_type, np.dtype):
            return np.issubdtype(obj_type, expected_type)
        # At this point we could try converting dtypes to pandas dtypes
        if isinstance(expected_type, pd.api.extensions.ExtensionDtype):
            return pd.api.types.is_dtype_equal(obj_type, expected_type)
        return False

    # Type checker that accounts for pandas dtypes
    def instanceof(obj, dtype):
        if isinstance(obj, dtype):
            return True
        obj_type = (
            type(obj) if not isinstance(obj, (pd.Series, np.ndarray)) else obj.dtype
        )
        return compare_types(obj_type, dtype)

    if instanceof(expr, dtype):
        return
    elif isinstance(expr, bodo.pandas.plan.ConstantExpression):
        if instanceof(expr.value, dtype):
            return
        else:
            expr_dtype = type(expr.value)
    elif isinstance(expr, ColRefExpression):
        if isinstance(expr.empty_data, pd.Series):
            expr_dtype = expr.empty_data.dtype
            if instanceof(expr.empty_data, dtype):
                return
        elif isinstance(expr.empty_data, pd.DataFrame):
            assert len(expr.empty_data.columns) == 1
            expr_dtype = expr.empty_data.dtypes[expr.empty_data.columns[0]]
            if compare_types(expr_dtype, dtype):
                return
        else:
            raise ValueError(
                f"Unsupported type of {expr_name}.empty_data:", type(expr.empty_data)
            )
    else:
        raise ValueError(f"Unsupported type of {expr_name}:", type(expr))

    raise ValueError(
        f"Expected {expr_name} ({type(expr)}) to hold datatype {str(dtype)}, instead was {expr_dtype}"
    )


def java_binop_to_python_expr(ctx, kind, op_name, op_exprs):
    """Convert a BodoSQL Java binary operator call to a DataFrame library expression."""

    left = op_exprs[0]

    # Calcite may add more than 2 operand for the same binary operator
    if len(op_exprs) > 2:
        right = java_binop_to_python_expr(ctx, kind, op_name, op_exprs[1:])
    else:
        right = op_exprs[1]

    SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind

    if kind.equals(SqlKind.PLUS):
        # TODO[BSE-5155]: support all BodoSQL data types in backend (including date/time)
        # TODO: upcast output to avoid overflow?
        out_empty = left.empty_data.iloc[:, 0] + right.empty_data.iloc[:, 0]
        expr = ArithOpExpression(out_empty, left, right, "__add__")
        return expr

    if kind.equals(SqlKind.MINUS):
        out_empty = left.empty_data.iloc[:, 0] - right.empty_data.iloc[:, 0]
        expr = ArithOpExpression(out_empty, left, right, "__sub__")
        return expr

    if kind.equals(SqlKind.TIMES):
        out_empty = left.empty_data.iloc[:, 0] * right.empty_data.iloc[:, 0]
        expr = ArithOpExpression(out_empty, left, right, "__mul__")
        return expr

    if kind.equals(SqlKind.DIVIDE):
        out_empty = left.empty_data.iloc[:, 0] / right.empty_data.iloc[:, 0]
        expr = ArithOpExpression(out_empty, left, right, "__truediv__")
        return expr

    # Comparison operators
    bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
    if kind.equals(SqlKind.EQUALS):
        return ComparisonOpExpression(bool_empty_data, left, right, operator.eq)

    if kind.equals(SqlKind.NOT_EQUALS):
        return ComparisonOpExpression(bool_empty_data, left, right, operator.ne)

    if kind.equals(SqlKind.LESS_THAN):
        return ComparisonOpExpression(bool_empty_data, left, right, operator.lt)

    if kind.equals(SqlKind.GREATER_THAN):
        return ComparisonOpExpression(bool_empty_data, left, right, operator.gt)

    if kind.equals(SqlKind.GREATER_THAN_OR_EQUAL):
        return ComparisonOpExpression(bool_empty_data, left, right, operator.ge)

    if kind.equals(SqlKind.LESS_THAN_OR_EQUAL):
        return ComparisonOpExpression(bool_empty_data, left, right, operator.le)

    if kind.equals(SqlKind.AND):
        return ConjunctionOpExpression(bool_empty_data, left, right, "__and__")

    if kind.equals(SqlKind.OR):
        return ConjunctionOpExpression(bool_empty_data, left, right, "__or__")

    if kind.equals(SqlKind.OTHER):
        if op_name == "||":  # string concatenation
            for op_expr in (left, right):
                ensure_type_of_expr(op_expr, "op_expr (|| arg)", str)

            separator = bodo.pandas.plan.ConstantExpression(
                left.empty_data,
                left.source,
                "",  # empty separator
            )
            return ArrowScalarFuncExpression(
                left.empty_data,
                [left, right, separator],
                "binary_join_element_wise",
                (),
            )

    raise NotImplementedError(f"Binary operator {kind.toString()} not supported yet")


def java_case_to_python_case(ctx, operands, input_plan):
    """Convert a BodoSQL Java CASE operator call to a DataFrame library CaseExpression.
    operands has the form [when1, then1, when2, then2, ..., else].
    """
    assert len(operands) >= 3, "CASE operator should have at least 3 operands"
    assert len(operands) % 2 == 1, "CASE operator should have an odd number of operands"
    when_expr = java_expr_to_python_expr(ctx, operands[0], input_plan)
    then_expr = java_expr_to_python_expr(ctx, operands[1], input_plan)

    if len(operands) > 3:
        else_expr = java_case_to_python_case(ctx, operands[2:], input_plan)
    else:
        else_expr = java_expr_to_python_expr(ctx, operands[2], input_plan)

    return CaseExpression(then_expr.empty_data, when_expr, then_expr, else_expr)


def java_join_to_python_join(ctx, java_join):
    """Convert a BodoSQL Java join plan to a Python join plan."""

    ctx.join_filter_info[java_join.getJoinFilterID()] = (
        java_join.getOriginalJoinFilterKeyLocations()
    )

    join_info = java_join.analyzeCondition()
    join_info_cls = join_info.getClass()
    field = join_info_cls.getField("nonEquiConditions")
    nonEquiConds = field.get(join_info)

    left_keys, right_keys = join_info.keys()
    key_indices = list(zip(left_keys, right_keys))
    join_type = JavaJoinTypeToDuckDB(java_join.getJoinType())
    force_broadcast = java_join.getBroadcastBuildSide()

    left_plan = java_plan_to_python_plan(ctx, java_join.getLeft())
    right_plan = java_plan_to_python_plan(ctx, java_join.getRight())

    empty_join_out = pd.concat([left_plan.empty_data, right_plan.empty_data], axis=1)
    empty_join_out.columns = java_join.getRowType().getFieldNames()

    if len(key_indices) > 0:
        planJoinOrCross = LogicalComparisonJoin(
            empty_join_out,
            left_plan,
            right_plan,
            join_type,
            key_indices,
            java_join.getJoinFilterID(),
            force_broadcast,
        )
    else:
        planJoinOrCross = LogicalCrossProduct(
            empty_join_out, left_plan, right_plan, force_broadcast
        )

    if len(nonEquiConds) == 0:
        return planJoinOrCross
    else:
        if java_join.getJoinType().toString() != "INNER":
            raise NotImplementedError(
                "Joins with non-equi conditions are only supported for inner joins in C++ backend currently"
            )
        non_equi_exprs = java_expr_to_python_expr(ctx, nonEquiConds[0], planJoinOrCross)
        # And all the conditions together with the first one above.
        for e in nonEquiConds[1:]:
            non_equi_exprs = ConjunctionOpExpression(
                non_equi_exprs.empty_data,
                non_equi_exprs,
                java_expr_to_python_expr(ctx, e, planJoinOrCross),
                "__and__",
            )
        # We convert a Calcite join with non-equi conditions into an equi join
        # plus a filter for a few reasons.  First, the dataframe join side does
        # not have an API that can create a join with non-equi conditions.
        # Those are only created by the duckdb optimizer so it doesn't hurt to
        # create the filter on top and left duckdb merge them.  Moreover, in
        # the CPU backend we convert joins with non-equi conditions back to a
        # join plus a filter so it is no big deal if duckdb doesn't merge the
        # join and this filter.  Also, this approach is a nicer match to the
        # architecture in this file and Calcite as the non-equi conditions
        # reference the joined table whereas our infrastructure would require
        # them to reference the build and probe side that is more tedious
        # work to do the mapping.
        planFilter = LogicalFilter(empty_join_out, planJoinOrCross, non_equi_exprs)
        return planFilter


def java_subplan_to_python_subplan(ctx, java_subplan):
    """Convert a BodoSQL Java subplan to a Python subplan."""

    if not hasattr(ctx, "subplan_cache"):
        ctx.subplan_cache = {}

    subplan_id = java_subplan.getCacheID()
    if subplan_id in ctx.subplan_cache:
        return ctx.subplan_cache[subplan_id]

    cached_plan = java_subplan.getCachedPlan()
    assert cached_plan.getClass().getSimpleName() == "CachedPlanInfo"
    subplan = java_plan_to_python_plan(ctx, cached_plan.getPlan())
    ctx.subplan_cache[subplan_id] = subplan
    return subplan


def java_rtjf_to_python_rtjf(ctx, java_plan):
    """Convert a BodoSQL Java runtime join filter plan to a Python runtime join filter
    plan.
    """
    input = java_plan_to_python_plan(ctx, java_plan.getInput())

    # Get join filter info
    # IDs of joins creating each filter
    filter_ids: list[int] = java_plan.getJoinFilterIDs()
    # Mapping columns of the join to the columns in the current table
    equality_filter_columns: list[list[int]] = java_plan.getEqualityFilterColumns()
    # Indicating for which of the columns is it the first filtering site
    equality_is_first_locations: list[list[bool]] = (
        java_plan.getEqualityIsFirstLocations()
    )

    # Zip tuples and sort all three lists by filter_ids
    sorted_filter_data = sorted(
        zip(filter_ids, equality_filter_columns, equality_is_first_locations),
        key=lambda x: x[0],
    )

    # Relocate filter columns based on original join filter key locations
    # See generateRuntimeJoinFilterCode() in BodoPhysicalRuntimeJoinFilter.kt
    new_filter_ids = []
    new_equality_filter_columns = []
    new_equality_is_first_locations = []
    for fid, eq_cols, is_first_cols in sorted_filter_data:
        if fid not in ctx.join_filter_info:
            raise ValueError(f"Join filter ID {fid} not found in join filter info")

        orig_key_locs = ctx.join_filter_info[fid]
        filter_cols = [-1] * len(eq_cols)
        is_first = [False] * len(is_first_cols)

        for loc_ind, key in enumerate(orig_key_locs):
            filter_cols[key] = eq_cols[loc_ind]
            is_first[key] = is_first_cols[loc_ind]

        new_filter_ids.append(fid)
        new_equality_filter_columns.append(filter_cols)
        new_equality_is_first_locations.append(is_first)

    return LogicalJoinFilter(
        input.empty_data,
        input,
        new_filter_ids,
        new_equality_filter_columns,
        new_equality_is_first_locations,
    )


def java_filter_to_python_filter(ctx, java_filter):
    """Convert a BodoSQL Java filter plan to a Python filter plan."""
    input_plan = java_plan_to_python_plan(ctx, java_filter.getInput())
    condition = java_expr_to_python_expr(ctx, java_filter.getCondition(), input_plan)
    return LogicalFilter(input_plan.empty_data, input_plan, condition)


def _is_interval_type(sql_type_name):
    """Check if a SqlTypeName is any interval subtype."""
    SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName
    return (
        sql_type_name.equals(SqlTypeName.INTERVAL_YEAR)
        or sql_type_name.equals(SqlTypeName.INTERVAL_MONTH)
        or sql_type_name.equals(SqlTypeName.INTERVAL_YEAR_MONTH)
        or sql_type_name.equals(SqlTypeName.INTERVAL_DAY)
        or sql_type_name.equals(SqlTypeName.INTERVAL_HOUR)
        or sql_type_name.equals(SqlTypeName.INTERVAL_MINUTE)
        or sql_type_name.equals(SqlTypeName.INTERVAL_SECOND)
        or sql_type_name.equals(SqlTypeName.INTERVAL_DAY_HOUR)
        or sql_type_name.equals(SqlTypeName.INTERVAL_DAY_MINUTE)
        or sql_type_name.equals(SqlTypeName.INTERVAL_DAY_SECOND)
        or sql_type_name.equals(SqlTypeName.INTERVAL_HOUR_MINUTE)
        or sql_type_name.equals(SqlTypeName.INTERVAL_HOUR_SECOND)
        or sql_type_name.equals(SqlTypeName.INTERVAL_MINUTE_SECOND)
    )


def _is_year_month_interval(sql_type_name):
    """Check if a SqlTypeName is a year/month interval subtype."""
    SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName
    return (
        sql_type_name.equals(SqlTypeName.INTERVAL_YEAR)
        or sql_type_name.equals(SqlTypeName.INTERVAL_MONTH)
        or sql_type_name.equals(SqlTypeName.INTERVAL_YEAR_MONTH)
    )


def java_literal_to_python_literal(ctx, java_literal, input_plan):
    """Convert a BodoSQL Java literal expression to a DataFrame library constant."""
    SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName
    lit_type = java_literal.getType()
    lit_type_name = lit_type.getSqlTypeName()

    # TODO[BSE-5156]: support all Calcite literal types

    if java_literal.getTypeName().equals(SqlTypeName.NULL):
        dummy_empty_data = pd.Series(
            dtype=pd.ArrowDtype(sql_type_to_pa_type(ctx, lit_type_name))
        )
        return NullExpression(dummy_empty_data, input_plan, 0)

    if lit_type_name.equals(SqlTypeName.DECIMAL):
        lit_type_scale = lit_type.getScale()
        val = java_literal.getValue()
        if lit_type_scale == 0:
            # Integer constants are represented as DECIMAL in Calcite
            dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            return ConstantExpression(dummy_empty_data, input_plan, int(val))
        else:
            # TODO: support proper decimal types in C++ backend
            dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            return ConstantExpression(dummy_empty_data, input_plan, float(val))

    if lit_type_name.equals(SqlTypeName.FLOAT):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float32()))
        return ConstantExpression(dummy_empty_data, input_plan, java_literal.getValue())

    if lit_type_name.equals(SqlTypeName.DOUBLE):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
        return ConstantExpression(dummy_empty_data, input_plan, java_literal.getValue())

    if lit_type_name.equals(SqlTypeName.BOOLEAN):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
        return ConstantExpression(dummy_empty_data, input_plan, java_literal.getValue())

    if lit_type_name.equals(SqlTypeName.CHAR):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.large_string()))
        return ConstantExpression(
            dummy_empty_data, input_plan, java_literal.getValue2()
        )

    if lit_type_name.equals(SqlTypeName.VARCHAR):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.large_string()))
        return ConstantExpression(
            dummy_empty_data, input_plan, java_literal.getValue2()
        )

    if lit_type_name.equals(SqlTypeName.DATE):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.date32()))
        # getValue2() returns an integer representing days since epoch
        val = pa.scalar(java_literal.getValue2(), pa.date32())
        return ConstantExpression(dummy_empty_data, input_plan, val)

    if lit_type_name.equals(SqlTypeName.TIMESTAMP):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.timestamp("ns")))
        # getValue2() returns an integer representing milliseconds since epoch
        val = pd.Timestamp(java_literal.getValue2(), unit="ms")
        return ConstantExpression(dummy_empty_data, input_plan, val)

    if _is_interval_type(lit_type_name):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.duration("ns")))
        if _is_year_month_interval(lit_type_name):
            # getValue() returns a BigDecimal representing total months
            months = int(java_literal.getValue())
            # cloudpickle can't serialize pyarrow month_day_nano_interval objects so we pass the values as integers and construct the interval in the C++ backend
            val = ("MonthDayNanoInterval", months, 0, 0)
        else:
            # Day/second subtypes: getValue2() returns a BigDecimal (Py4J
            # converts to decimal.Decimal) representing milliseconds.
            millis = float(str(java_literal.getValue2()))
            nanos = int(millis * 1_000_000)
            # cloudpickle can't serialize pyarrow month_day_nano_interval objects so we pass the values as integers and construct the interval in the C++ backend
            val = ("MonthDayNanoInterval", 0, 0, nanos)
        return ConstantExpression(dummy_empty_data, input_plan, val)

    if (
        lit_type_name.equals(SqlTypeName.TINYINT)
        or lit_type_name.equals(SqlTypeName.SMALLINT)
        or lit_type_name.equals(SqlTypeName.INTEGER)
        or lit_type_name.equals(SqlTypeName.BIGINT)
    ):
        dummy_empty_data = pd.Series(
            dtype=pd.ArrowDtype(sql_type_to_pa_type(ctx, lit_type_name))
        )
        return ConstantExpression(
            dummy_empty_data, input_plan, java_literal.getValue2()
        )

    raise NotImplementedError(
        f"Literal type {lit_type_name.toString()} not supported yet"
    )


def is_int_type(java_type):
    """Check if a Calcite type is an integer type."""
    SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName
    type_name = java_type.getSqlTypeName()
    return (
        type_name.equals(SqlTypeName.TINYINT)
        or type_name.equals(SqlTypeName.SMALLINT)
        or type_name.equals(SqlTypeName.INTEGER)
        or type_name.equals(SqlTypeName.BIGINT)
    )


def gen_plan_via_bodo_dataframe(func, *args, **kwargs):
    """Generate a Python plan for this module by wrapping input plans with
    dataframes/series and calling a func parameter that is written in
    regular Pandas/Bodo DataFrame library syntax.  Finally, convert the
    dataframe returned by that function to a plan."""
    # Wrap any LazyPlans in the args or kwargs as dataframes/series.
    args = [arg if not isinstance(arg, LazyPlan) else wrap_plan(arg) for arg in args]
    kwargs = {
        k: (arg if not isinstance(arg, LazyPlan) else wrap_plan(arg))
        for k, arg in kwargs.items()
    }
    # Call the func to generate a new plan.
    output_dataframe = func(*args, **kwargs)
    # Extract that plan from the returned dataframe.
    assert isinstance(output_dataframe, bodo.pandas.DataFrame)
    assert output_dataframe.is_lazy_plan()
    return output_dataframe._plan


def java_agg_to_python_agg(ctx, java_plan):
    """Convert a BodoSQL Java aggregation plan to a Python aggregation plan."""
    from bodo.pandas.groupby import GroupbyAggFunc, _get_agg_output_type

    keys = list(java_plan.getGroupSet().toList())

    input_plan = java_plan_to_python_plan(ctx, java_plan.getInput())

    exprs = []
    out_types = [input_plan.pa_schema.field(k).type for k in keys]
    aggCallList = java_plan.getAggCallList()
    if len(aggCallList) == 0:
        # If no aggregation expressions then use distinct instead.

        names = list(java_plan.getRowType().getFieldNames())
        new_schema = pa.schema([pa.field(name, t) for name, t in zip(names, out_types)])
        empty_out_data = arrow_to_empty_df(new_schema)

        exprs = make_col_ref_exprs(keys, input_plan)
        plan = LogicalDistinct(
            empty_out_data,
            input_plan,
            exprs,
        )
        return plan

    # calcite supports a literal "aggregation" function but Bodo backend doesn't.
    # So capture those functions here and treat them as projections later.
    literal_aggs = []
    for aggIndex, func in enumerate(aggCallList):
        if func.hasFilter():
            raise NotImplementedError("Filtered aggregations are not supported yet")
        func_name = _agg_to_func_name(func)
        arg_cols = list(func.getArgList())
        if func_name == "size":
            assert len(arg_cols) == 0, (
                "Size aggregations with non-zero arg len not supported"
            )
            out_type = pa.int64()
        elif func_name == "count":
            assert len(arg_cols) == 1, (
                "Only single-argument count aggregations are supported"
            )
            out_type = pa.int64()
        elif func_name == "nunique":
            assert len(arg_cols) == 1, (
                "Only single-argument nunique aggregations are supported"
            )
            out_type = pa.int64()
        elif func_name in [
            "sum",
            "max",
            "min",
            "std",
            "mean",
            "var",
            "var_pop",
            "skew",
            "kurtosis",
        ]:
            assert len(arg_cols) == 1, (
                f"Only single-argument {func_name} aggregations are supported"
            )
            in_type = input_plan.pa_schema.field(arg_cols[0]).type
            out_type = _get_agg_output_type(
                GroupbyAggFunc("dummy", func_name), in_type, "dummy"
            )
        elif func_name in ["boolor_agg", "booland_agg", "boolxor_agg"]:
            assert len(arg_cols) == 1, (
                f"Only single-argument {func_name} aggregations are supported"
            )
            out_type = pa.bool_()
        elif func_name == "literal_agg":
            rexlist = func.getClass().getDeclaredField("rexList").get(func)
            assert len(rexlist) == 1
            literal_for_literal_agg = java_literal_to_python_literal(
                ctx, rexlist.get(0), input_plan
            )
            # Save column index where this literal aggregation should appear
            # and the value of the literal.
            literal_aggs.append((aggIndex + len(keys), literal_for_literal_agg))
            continue
        else:
            raise NotImplementedError(
                f"java_agg_to_python_agg: aggregation {func_name} not supported yet"
            )

        out_types.append(out_type)
        exprs.append(
            AggregateExpression(
                pd.Series([], dtype=pd.ArrowDtype(out_type)),
                input_plan,
                func_name,
                None,
                arg_cols,
                False,
            )
        )

    # All column names of the desired output in order.
    names = list(java_plan.getRowType().getFieldNames())
    agg_names = []
    # Column indices we should skip to start with since they are literal
    # aggregations to be added later.
    skipped_literal_indices = [x[0] for x in literal_aggs]
    # Get just the column names that Bodo will do as a true aggregation.
    for nindex in range(len(names)):
        if nindex not in skipped_literal_indices:
            agg_names.append(names[nindex])
    if len(agg_names) > len(keys):
        # There is some non-literal_agg aggregate.
        new_schema = pa.schema(
            [pa.field(name, t) for name, t in zip(agg_names, out_types)]
        )
        empty_out_data = arrow_to_empty_df(new_schema)

        # Do the real aggregations.
        plan = LogicalAggregate(
            empty_out_data,
            input_plan,
            keys,
            exprs,
        )

        # If there is some literal_agg we have to deal with.
        if len(agg_names) != len(names):

            def add_lits_to_agg(df, names, literal_aggs):
                # Add in the literal_agg columns.
                for literal_agg in literal_aggs:
                    df[names[literal_agg[0]]] = literal_agg[1].value
                # Reorder to the originally calculated order.
                return df[names]

            plan = gen_plan_via_bodo_dataframe(
                add_lits_to_agg, plan, names, literal_aggs
            )
    else:
        # There were no non-literal aggregations so just select the keys and
        # drop_duplicates and then add on the literal aggregation columns.
        def select_keys_lits(plan, keys, names, literal_aggs):
            cols = plan.columns[keys].tolist()
            key_df = plan[cols].drop_duplicates()
            for literal_agg in literal_aggs:
                key_df[names[literal_agg[0]]] = literal_agg[1].value
            return key_df

        plan = gen_plan_via_bodo_dataframe(
            select_keys_lits, input_plan, keys, names, literal_aggs
        )
    return plan


def _agg_to_func_name(func):
    """Map a Calcite aggregation to a groupby function name."""
    agg = func.getAggregation()
    SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind
    kind = agg.getKind()
    agg_name = agg.getName()

    argList = func.getArgList()

    # TODO[BSE-5163]: support SUM0 initialization properly
    if kind.equals(SqlKind.SUM) or kind.equals(SqlKind.SUM0):
        return "sum"

    if kind.equals(SqlKind.COUNT) and len(argList) == 0:
        return "size"

    if kind.equals(SqlKind.COUNT) and len(argList) == 1:
        return "count"

    if kind.equals(SqlKind.MAX) and len(argList) == 1:
        return "max"

    if kind.equals(SqlKind.MIN) and len(argList) == 1:
        return "min"

    if kind.equals(SqlKind.AVG) and len(argList) == 1:
        return "mean"

    if kind.equals(SqlKind.STDDEV_SAMP) and len(argList) == 1:
        return "std"

    if kind.equals(SqlKind.LITERAL_AGG) and len(argList) == 0:
        return "literal_agg"

    if kind.equals(SqlKind.OTHER):
        if agg_name == "BOOLOR_AGG":
            return "boolor_agg"
        if agg_name == "BOOLAND_AGG":
            return "booland_agg"
        if agg_name == "BOOLXOR_AGG":
            return "boolxor_agg"
        raise NotImplementedError(f"Aggregation {agg_name} not supported yet")

    if kind.equals(SqlKind.OTHER_FUNCTION):
        # Normalize name for matching
        name = agg_name.upper() if agg_name is not None else ""

        if name == "VARIANCE_SAMP":
            return "var"
        if name == "VARIANCE_POP":
            return "var_pop"
        if name == "SKEW":
            return "skew"
        if name == "KURTOSIS":
            return "kurtosis"

        details = ""

        # If the agg object exposes more metadata, try to print it for debugging
        try:
            cls_name = agg.getClass().getName()
        except Exception:
            cls_name = "<unknown-class>"

        # Try to extract an underlying function object or identifier if present
        extra_info = {}
        try:
            # Many Calcite SqlAggFunction subclasses have methods like getFunction or getIdentifier
            if hasattr(agg, "getFunction"):
                try:
                    func_obj = agg.getFunction()
                    extra_info["function_class"] = (
                        func_obj.getClass().getName()
                        if func_obj is not None
                        else "null"
                    )
                except Exception:
                    extra_info["function_class"] = "<unreadable>"
            if hasattr(agg, "getIdentifier"):
                try:
                    ident = agg.getIdentifier()
                    extra_info["identifier"] = str(ident)
                except Exception:
                    extra_info["identifier"] = "<unreadable>"
            if hasattr(agg, "getOperandTypes"):
                try:
                    extra_info["operand_types"] = str(agg.getOperandTypes())
                except Exception:
                    extra_info["operand_types"] = "<unreadable>"
        except Exception:
            # ignore reflection failures
            pass

        # Print a helpful debug dump to stderr so you can see what Calcite provided
        details += "DEBUG: Unmapped aggregation encountered in _agg_to_func_name()\n"
        details += f"  agg_name: {agg_name}\n"
        details += f"  kind: {kind.toString() if kind is not None else 'null'}\n"
        details += f"  agg_class: {cls_name}\n"
        if extra_info:
            for k, v in extra_info.items():
                details += f"  {k}: {v}\n"

        raise NotImplementedError(
            f"Aggregation {agg_name} (class={cls_name}) not supported yet\n{details}"
        )

    raise NotImplementedError(f"Aggregation {kind.toString()} not supported yet")


def java_sort_to_python_sort(ctx, java_plan):
    """Convert a BodoSQL Java sort plan to a Python sort plan."""

    if java_plan.getOffset() is not None:
        raise NotImplementedError("OFFSET in sort not supported yet")

    input_plan = java_plan_to_python_plan(ctx, java_plan.getInput())

    sort_collations = java_plan.getCollation().getFieldCollations()
    key_col_inds = []
    ascending = []
    na_position = []
    for collation in sort_collations:
        field_index = collation.getFieldIndex()
        descending = collation.getDirection().isDescending()
        is_nulls_first = gateway.jvm.com.bodosql.calcite.adapter.bodo.BodoPhysicalSort.Companion.isNullsFirst(
            collation
        )
        key_col_inds.append(field_index)
        ascending.append(not descending)
        na_position.append(is_nulls_first)

    limit = java_plan.getFetch()
    if limit is None:
        return LogicalOrder(
            input_plan.empty_data,
            input_plan,
            ascending,
            na_position,
            key_col_inds,
            input_plan.pa_schema,
        )
    else:
        limit_expr = java_expr_to_python_expr(ctx, limit, input_plan)
        return LogicalTopN(
            input_plan.empty_data,
            input_plan,
            ascending,
            na_position,
            key_col_inds,
            input_plan.pa_schema,
            limit_expr.value,
            0,
        )


def java_values_to_python_values(ctx, java_plan):
    """Convert a BodoSQL Java BodoPhysicalValues plan to a Python DataFrame read plan."""
    rows = java_plan.getTuples()
    row_type = java_plan.getRowType()

    data = []
    for row in rows:
        data.append([java_literal_to_python_literal(ctx, e, None).value for e in row])

    pa_schema = pa.schema(
        [java_field_to_pa_field(ctx, f) for f in row_type.getFieldList()]
    )

    df = pd.DataFrame()
    for i, name in enumerate(pa_schema.names):
        df[name] = pd.Series(
            [data[j][i] for j in range(len(data))],
            dtype=pd.ArrowDtype(pa_schema.field(i).type),
        )

    return bd.from_pandas(df)._plan


def java_field_to_pa_field(ctx, java_field):
    """Convert a Calcite RelDataTypeField to a PyArrow field."""
    name = java_field.getName()
    java_type = java_field.getType()
    type_name = java_type.getSqlTypeName()

    return pa.field(name, sql_type_to_pa_type(ctx, type_name))


def sql_type_to_pa_type(ctx, sql_type_name):
    """Convert a Calcite SqlTypeName to a PyArrow data type."""
    SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName

    if sql_type_name.equals(SqlTypeName.TINYINT):
        return pa.int8()
    if sql_type_name.equals(SqlTypeName.SMALLINT):
        return pa.int16()
    if sql_type_name.equals(SqlTypeName.INTEGER):
        return pa.int32()
    if sql_type_name.equals(SqlTypeName.BIGINT):
        return pa.int64()
    if sql_type_name.equals(SqlTypeName.FLOAT):
        return pa.float32()
    if sql_type_name.equals(SqlTypeName.DOUBLE):
        return pa.float64()
    if sql_type_name.equals(SqlTypeName.VARCHAR):
        return pa.large_string()
    if sql_type_name.equals(SqlTypeName.VARBINARY):
        return pa.large_binary()
    if sql_type_name.equals(SqlTypeName.DATE):
        return pa.date32()
    if sql_type_name.equals(SqlTypeName.TIMESTAMP):
        return pa.timestamp("ns")
    if sql_type_name.equals(SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE):
        # BodoSQL uses UTC if timezone is not specified
        tz = ctx.default_tz if ctx.default_tz is not None else "UTC"
        return pa.timestamp("ns", tz=tz)
    if _is_interval_type(sql_type_name):
        return pa.duration("ns")
    if sql_type_name.equals(SqlTypeName.BOOLEAN):
        return pa.bool_()
    if sql_type_name.equals(SqlTypeName.CHAR):
        return pa.large_string()
    if sql_type_name.equals(SqlTypeName.TIME):
        return pa.time64("ns")

    raise NotImplementedError(f"SQL type {sql_type_name.toString()} not supported yet")


def visit_iceberg_node(java_plan, read_info):
    """Visit Iceberg-related plan nodes to extract read information like filters.
    For example:
    CombineStreamsExchange
        IcebergToBodoPhysicalConverter
            IcebergFilter(condition=[>($3, 3.1E0)])
                IcebergTableScan(...)
    """
    java_class_name = java_plan.getClass().getSimpleName()

    if java_class_name == "IcebergTableScan":
        read_info.scan_node = java_plan
        return

    if java_class_name == "IcebergFilter":
        input = java_plan.getInput()
        if read_info.filters is None:
            read_info.filters = []
        read_info.filters.append(java_plan.getCondition())
        visit_iceberg_node(input, read_info)
        return

    if java_class_name == "IcebergProject":
        # Projects may reorder columns, so we need to update the column mapping.
        # See IcebergToBodoPhysicalConverter.kt
        new_colmap = []
        projs = java_plan.getProjects()
        for ind in read_info.colmap:
            proj = projs[ind]
            if proj.getClass().getSimpleName() != "RexInputRef":
                raise NotImplementedError(
                    "IcebergProject with expressions not supported yet"
                )
            new_colmap.append(proj.getIndex())

        read_info.colmap = new_colmap
        input = java_plan.getInput()
        visit_iceberg_node(input, read_info)
        return

    if java_class_name == "IcebergSort":
        limit = java_plan.getFetch()
        if limit is not None:
            assert limit.getClass().getSimpleName() == "RexLiteral", (
                "Only literal LIMITs are supported in IcebergSort"
            )
            limit = java_expr_to_pyiceberg_expr(limit, [])
            read_info.limit = (
                limit if read_info.limit is None else min(read_info.limit, limit)
            )
        input = java_plan.getInput()
        visit_iceberg_node(input, read_info)
        return

    raise NotImplementedError(
        f"Iceberg plan node {java_class_name} not supported yet in visit_iceberg_node"
    )


def generate_iceberg_read(read_info):
    """Generate a Python plan for reading Iceberg table with the given read info."""
    scan_node = read_info.scan_node
    catalog_table = scan_node.getCatalogTable()
    catalog = catalog_table.getCatalog()
    # TODO: support other catalog types
    if catalog.getClass().getSimpleName() != "FileSystemCatalog":
        raise NotImplementedError(
            "Only FileSystemCatalog is supported in IcebergTableScan in C++ backend"
        )

    # Get table info
    full_table_path = catalog_table.getFullPath()
    schema_path = catalog_table.getParentFullPath()
    field_names = scan_node.deriveRowType().getFieldNames()

    row_filter = get_pyiceberg_row_filter(read_info.filters, field_names)
    read_fields = [field_names[i] for i in read_info.colmap]

    # Get file system path
    file_path = catalog.schemaPathToFilePath(schema_path)
    uri = file_path.toUri()
    path_str = uri.getRawPath()

    df = bd.read_iceberg(
        # path_str has the schema in it so it's not needed in table id
        # TODO: update when supporting other catalog types
        full_table_path[-1],
        location=path_str,
        row_filter=row_filter,
        selected_fields=read_fields,
        limit=read_info.limit,
    )
    return df._plan


def get_pyiceberg_row_filter(filters, field_names):
    """Convert SQL filters to a PyIceberg filter expression for
    bodo.pandas.read_iceberg()
    """
    if filters is None or len(filters) == 0:
        return None

    op_exprs = [java_expr_to_pyiceberg_expr(o, field_names) for o in filters]

    if len(op_exprs) == 1:
        return op_exprs[0]

    # AND all filters
    SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind
    return java_binop_to_pyiceberg_expr(SqlKind.AND, op_exprs)


def java_expr_to_pyiceberg_expr(java_expr, field_names):
    """Convert a BodoSQL Java expression to a PyIceberg expression"""
    import pyiceberg.expressions as pie

    java_class_name = java_expr.getClass().getSimpleName()

    if java_class_name == "RexInputRef":
        col_index = java_expr.getIndex()
        return pie.Reference(field_names[col_index])

    if java_class_name == "RexCall":
        return java_call_to_pyiceberg_call(java_expr, field_names)

    if java_class_name == "RexLiteral":
        return java_literal_to_pyiceberg_literal(java_expr)

    raise NotImplementedError(
        f"Expression {java_class_name} not supported yet in java_expr_to_pyiceberg_expr"
    )


def java_call_to_pyiceberg_call(java_call, field_names):
    """Convert a BodoSQL Java call expression to a PyIceberg expression"""
    import pyiceberg.expressions as pie

    op = java_call.getOperator()
    operator_class_name = op.getClass().getSimpleName()

    if operator_class_name in ("SqlMonotonicBinaryOperator", "SqlBinaryOperator"):
        operands = java_call.getOperands()
        # Calcite may add more than 2 operand for the same binary operator
        op_exprs = [java_expr_to_pyiceberg_expr(o, field_names) for o in operands]
        kind = op.getKind()
        return java_binop_to_pyiceberg_expr(kind, op_exprs)

    if operator_class_name == "SqlCastFunction" and len(java_call.getOperands()) == 1:
        operand = java_call.getOperands()[0]
        operand_type = operand.getType()
        target_type = java_call.getType()
        SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName
        # TODO[BSE-5154]: support all Calcite casts

        if target_type.getSqlTypeName().equals(SqlTypeName.DECIMAL) and is_int_type(
            operand_type
        ):
            # Cast of int to DECIMAL is unnecessary in C++ backend
            return java_expr_to_pyiceberg_expr(operand, field_names)

        if operand_type.getSqlTypeName().equals(
            SqlTypeName.VARCHAR
        ) and target_type.getSqlTypeName().equals(SqlTypeName.VARCHAR):
            # No-op cast of VARCHAR (could be different lengths but sometimes equal
            # which seems like a Calcite gap)
            return java_expr_to_pyiceberg_expr(operand, field_names)

    if (
        operator_class_name == "SqlPostfixOperator"
        and len(java_call.getOperands()) == 1
    ):
        operands = java_call.getOperands()
        input = java_expr_to_pyiceberg_expr(operands[0], field_names)
        kind = op.getKind()
        SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind

        if kind.equals(SqlKind.IS_NOT_NULL):
            return pie.NotNull(input)

        if kind.equals(SqlKind.IS_NULL):
            return pie.IsNull(input)

    raise NotImplementedError(
        f"Call operator {operator_class_name} for pyiceberg not supported yet: "
        + java_call.toString()
    )


def java_binop_to_pyiceberg_expr(kind, op_exprs):
    """Convert a BodoSQL Java binary operator call to a DataFrame library expression."""
    import pyiceberg.expressions as pie

    left = op_exprs[0]

    # Calcite may add more than 2 operand for the same binary operator
    if len(op_exprs) > 2:
        right = java_binop_to_pyiceberg_expr(kind, op_exprs[1:])
    else:
        right = op_exprs[1]

    SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind

    # Comparison operators
    if kind.equals(SqlKind.EQUALS):
        return pie.EqualTo(left, right)

    if kind.equals(SqlKind.NOT_EQUALS):
        return pie.NotEqualTo(left, right)

    if kind.equals(SqlKind.LESS_THAN):
        return pie.LessThan(left, right)

    if kind.equals(SqlKind.GREATER_THAN):
        return pie.GreaterThan(left, right)

    if kind.equals(SqlKind.GREATER_THAN_OR_EQUAL):
        return pie.GreaterThanOrEqual(left, right)

    if kind.equals(SqlKind.LESS_THAN_OR_EQUAL):
        return pie.LessThanOrEqual(left, right)

    if kind.equals(SqlKind.AND):
        left = _ensure_pyiceberg_non_ref_expr(left)
        right = _ensure_pyiceberg_non_ref_expr(right)
        return pie.And(left, right)

    if kind.equals(SqlKind.OR):
        left = _ensure_pyiceberg_non_ref_expr(left)
        right = _ensure_pyiceberg_non_ref_expr(right)
        return pie.Or(left, right)

    raise NotImplementedError(
        f"Binary operator {kind.toString()} not supported yet in java_binop_to_pyiceberg_expr"
    )


def _ensure_pyiceberg_non_ref_expr(expr):
    """PyIceberg cannot handle "loose" References in AND/OR expressions so this function
    converts them to EqualTo(expr, True) expressions.
    Example query:
    select * from \"my_schema\".\"sss\".\"table1\" where \"four\" > 3.1 and \"three\"
    """
    import pyiceberg.expressions as pie

    if isinstance(expr, pie.Reference):
        return pie.EqualTo(expr, True)

    return expr


def java_literal_to_pyiceberg_literal(java_literal):
    """Convert a BodoSQL Java literal expression to a constant to use in PyIceberg
    expressions.
    """
    SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName
    lit_type_name = java_literal.getTypeName()
    lit_type = java_literal.getType()

    # TODO[BSE-5156]: support all Calcite literal types

    if lit_type_name.equals(SqlTypeName.DECIMAL):
        lit_type_scale = lit_type.getScale()
        val = java_literal.getValue()
        if lit_type_scale == 0:
            return int(val)
        else:
            return val

    if lit_type_name.equals(SqlTypeName.DOUBLE):
        return java_literal.getValue()

    if lit_type_name.equals(SqlTypeName.CHAR):
        return java_literal.getValue2()

    if lit_type_name.equals(SqlTypeName.DATE):
        # getValue2() returns an integer representing days since epoch
        val = pa.scalar(java_literal.getValue2(), pa.date32())
        return val

    raise NotImplementedError(
        f"Literal type {lit_type_name.toString()} not supported yet in java_literal_to_pyiceberg_literal"
    )


def JavaJoinTypeToDuckDB(java_join_type):
    from bodo.ext import plan_optimizer

    JoinRelType = gateway.jvm.org.apache.calcite.rel.core.JoinRelType

    if java_join_type.equals(JoinRelType.INNER):
        return plan_optimizer.CJoinType.INNER

    if java_join_type.equals(JoinRelType.LEFT):
        return plan_optimizer.CJoinType.LEFT

    if java_join_type.equals(JoinRelType.RIGHT):
        return plan_optimizer.CJoinType.RIGHT

    if java_join_type.equals(JoinRelType.FULL):
        return plan_optimizer.CJoinType.OUTER

    raise NotImplementedError(
        f"Join type {java_join_type.toString()} not supported yet"
    )
