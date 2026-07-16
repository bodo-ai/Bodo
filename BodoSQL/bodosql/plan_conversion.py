from __future__ import annotations

import decimal
import operator
import re
import zoneinfo
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import py4j
import pyarrow as pa
import pyarrow.compute as pc

import bodo
import bodo.pandas as bd
import bodosql
from bodo.pandas.iceberg_utils import (
    JoinFilterInfo,
    build_iceberg_read_plan,
)
from bodo.pandas.plan import (
    AggregateExpression,
    ArithOpExpression,
    ArrowScalarFuncExpression,
    CaseExpression,
    CastExpression,
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
    LogicalSetOperation,
    LogicalTopN,
    NullExpression,
    PythonScalarFuncExpression,
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
    "DAYOFWEEKISO": "day_of_week",
    "WEEKDAY": "day_of_week",
    "YEAROFWEEK": "iso_year",
    "YEAROFWEEKISO": "iso_year",
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
    join_filter_info: JoinFilterInfo = None
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
        visit_iceberg_node(ctx, input, read_info)
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
        input_python_plan = java_plan_to_python_plan(ctx, java_plan.getInput())
        join_info = java_rtjf_to_join_info(ctx, java_plan)
        return generate_runtime_join_filter(join_info, input_python_plan)

    if java_class_name == "BodoPhysicalFilter":
        return java_filter_to_python_filter(ctx, java_plan)

    if java_class_name == "BodoPhysicalAggregate":
        # TODO: support grouping sets
        if java_plan.usesGroupingSets():
            raise NotImplementedError(
                "BodoPhysicalAggregate with grouping sets is not supported in C++ backend yet"
            )
        return java_agg_to_python_agg(ctx, java_plan)

    if java_class_name == "BodoPhysicalSort":
        return java_sort_to_python_sort(ctx, java_plan)

    if java_class_name == "BodoPhysicalValues":
        return java_values_to_python_values(ctx, java_plan)

    if java_class_name == "BodoPhysicalCachedSubPlan":
        return java_subplan_to_python_subplan(ctx, java_plan)

    if java_class_name == "BodoPhysicalTableCreate":
        return java_table_create_to_python(ctx, java_plan)

    if java_class_name == "BodoPhysicalUnion":
        return java_union_to_python_union(ctx, java_plan)

    raise NotImplementedError(f"Plan node {java_class_name} not supported yet")


def java_union_to_python_union(ctx, java_plan):
    input_plans = [java_plan_to_python_plan(ctx, x) for x in java_plan.getInputs()]
    plan = LogicalSetOperation(
        input_plans[-2].empty_data, input_plans[-1], input_plans[-2], "union all"
    )
    for i in range(len(input_plans) - 3, -1, -1):
        plan = LogicalSetOperation(
            input_plans[i].empty_data, plan, input_plans[i], "union all"
        )
    return plan


def java_table_create_to_python(ctx, java_plan):
    input_plan = java_plan_to_python_plan(ctx, java_plan.getInput())
    return ctx.NewTable(input_plan, java_plan)


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

    def timestamp_from_parts(
        year_expr,
        month_expr,
        day_expr,
        hour_expr=None,
        minute_expr=None,
        second_expr=None,
        nanosecond_expr=None,
    ):
        """
        Create a timestamp expression from the provided part expressions
        (instances of bodo.pandas.plan.Expression).

        If the inputs are within the normal ranges, the timestamp is constructed
        as expected. However, the inputs (aside from the year) can also be
        outside of the usual ranges in which they would appear in a timestamp.
        At a high level, the resulting timestamp from this function will simply
        be the sum of the parts.

        More precisely, if month_expr = M, we add M-1 months after January of the
        given year. Therefore, if M = 0, we subtract a month to get December of the
        previous year, and if M = -1, we would go back two months.
        The same rule applies to day_expr. We add day_expr - 1 days to the 1st of
        the given month.
        Adding the time components hour_expr, minute_expr, second_expr, and
        nanosecond_expr is more intuitive because they start at zero.

        Specifying the hour, minute, seconds, and nanosecond are optional,
        and will default to 0 if not provided.

        Any float inputs will be rounded to integers. Float inputs that cannot
        fit in int64 will cause a NULL timestamp to be emitted.

        The result will be timestamp[s] unless nanoseconds are provided, where we
        return timestamp[ns]. In theory we can represent more years in an int64
        timestamp this way, although the C++ backend may need better support for
        returning timestamps of lower than ns resolution.
        """

        """
        Since Arrow does not have a timestamp_from_parts function,
        here is an overview of our approach:

        We have a couple options to create a timestamp. One is to
        calculate epoch time and then cast directly from an integer
        to a timestamp. This method seems impractical
        because you have to be conscious about days per month, leap
        years, etc. The other option is to use Arrow's strptime, to
        parse a formatted string as a timestamp.

        Unfortunately we can't call strptime right away, because the given
        timestamp components could be out of range. We could manully 
        accumulate the components one by one using mod arithmetic to
        calculate the effective timestamp part values, but it would be
        tedious and potentially inefficient.
        
        Therefore we only compute the final year and month values beforehand
        to ensure calendar awareness. Then we use those values with the first
        day of that month to create a timestamp type via strptime. At this point,
        we need to add the extra days, hours, minutes, seconds, and nanoseconds
        we have yet to account for. We convert those to a common unit (seconds
        or nanoseconds) and cast to a duration type before adding to our
        year-month timestamp.
        """

        # Arrow currently lacks a function to make a date or timestamp from parts:
        # https://github.com/apache/arrow/issues/49514.
        # There is some possibility of it being added in the future,
        # so we can revisit this strptime approach then.

        int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
        float_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
        bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
        string_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.string()))

        max_int64 = ConstantExpression(
            int_empty_data, input_plan, np.iinfo(np.int64).max
        )
        min_int64 = ConstantExpression(
            int_empty_data, input_plan, np.iinfo(np.int64).min
        )

        # If any of the inputs are floats, we will round to the nearest integer.
        # This is what the JIT implementation seems to do, even though float input isn't explicitly allowed in Snowflake.
        # See `construct_timestamp` in BodoSQL/bodosql/kernels/datetime_array_kernels.py.
        def int_part_expr(part_expr, part_expr_name):
            ensure_type_of_expr(part_expr, part_expr_name, (int, float))
            part_expr_dtype = get_expr_dtype(part_expr, part_expr_name)
            if compare_types(part_expr_dtype, float):
                part_expr = ArrowScalarFuncExpression(
                    float_empty_data,
                    [part_expr],
                    "round",
                    (0, "half_towards_infinity"),
                )

                # The BodoSQL docs say we should return NULL when any input cannot be converted to int64
                part_too_large = ComparisonOpExpression(
                    bool_empty_data, part_expr, max_int64, operator.gt
                )
                part_too_small = ComparisonOpExpression(
                    bool_empty_data, part_expr, min_int64, operator.lt
                )
                part_out_of_bounds = ConjunctionOpExpression(
                    bool_empty_data, part_too_large, part_too_small, "__or__"
                )
                part_expr = CaseExpression(
                    float_empty_data,
                    part_out_of_bounds,
                    NullExpression(float_empty_data, input_plan, 0),
                    part_expr,
                )

                part_expr = CastExpression(int_empty_data, part_expr)
            return part_expr

        year_expr = int_part_expr(year_expr, "year_expr")
        month_expr = int_part_expr(month_expr, "month_expr")
        day_expr = int_part_expr(day_expr, "day_expr")

        if hour_expr:
            hour_expr = int_part_expr(hour_expr, "hour_expr")
        if minute_expr:
            minute_expr = int_part_expr(minute_expr, "minute_expr")
        if second_expr:
            second_expr = int_part_expr(second_expr, "second_expr")
            # Ensure seconds are int64 so the result of the later multiplication doesn't
            # end up being int32 internally, potentially leading to overflow and the wrong answer.
            second_expr = CastExpression(int_empty_data, second_expr)
        if nanosecond_expr:
            nanosecond_expr = int_part_expr(nanosecond_expr, "nanosecond_expr")

        one_expr = ConstantExpression(int_empty_data, input_plan, 1)
        zero_expr = ConstantExpression(int_empty_data, input_plan, 0)
        twelve_expr = ConstantExpression(int_empty_data, input_plan, 12)

        # Timestamp components (aside from the year) can be outside the usual ranges
        # or even negative, so we must account for this and adjust the constructed timestamp.

        # Calculate the year and month manually to ensure calendar awareness.

        # Calculate month component.
        # If (month-1) % 12 is 0 or positive, month component is (month-1) % 12 + 1.
        # If (month-1) % 12 is negative, month component is 13 + (month-1) % 12.
        month_minus_one = ArithOpExpression(
            int_empty_data, month_expr, one_expr, "__sub__"
        )
        # We rely on bodo_mod returning a negative remainder when (month-1) is negative
        month_remainder = ArithOpExpression(
            int_empty_data, month_minus_one, twelve_expr, "__mod__"
        )
        month_remainder_negative = ComparisonOpExpression(
            bool_empty_data, month_remainder, zero_expr, operator.lt
        )
        month_num_to_add = CaseExpression(
            int_empty_data,
            month_remainder_negative,
            ConstantExpression(int_empty_data, input_plan, 13),
            one_expr,
        )
        month_component = ArithOpExpression(
            int_empty_data, month_remainder, month_num_to_add, "__add__"
        )

        # Calculate year component.
        # If month is positive, year component is year + (month-1) // 12.
        # If month is 0 or negative, year component is year + (month) // 12 - 1.
        # Here we rely on DuckDB's integer division to truncate towards zero.
        month_positive = ComparisonOpExpression(
            bool_empty_data, month_expr, zero_expr, operator.gt
        )
        month_quotient = ArithOpExpression(
            int_empty_data, month_expr, twelve_expr, "__floordiv__"
        )
        month_quotient_minus_one = ArithOpExpression(
            int_empty_data, month_quotient, one_expr, "__sub__"
        )
        month_minus_one_quotient = ArithOpExpression(
            int_empty_data, month_minus_one, twelve_expr, "__floordiv__"
        )
        year_offset = CaseExpression(
            int_empty_data,
            month_positive,
            month_minus_one_quotient,
            month_quotient_minus_one,
        )
        year_component = ArithOpExpression(
            int_empty_data, year_expr, year_offset, "__add__"
        )

        # Concatenate date components to get a string that can be parsed by strptime
        year_string = CastExpression(string_empty_data, year_component)
        year_string = ArrowScalarFuncExpression(
            string_empty_data, [year_string], "utf8_lpad", (4, "0")
        )
        month_string = CastExpression(string_empty_data, month_component)
        month_string = ArrowScalarFuncExpression(
            string_empty_data, [month_string], "utf8_lpad", (2, "0")
        )
        day_string = ConstantExpression(string_empty_data, input_plan, "01")
        separator = ConstantExpression(string_empty_data, input_plan, "-")
        date_string = ArrowScalarFuncExpression(
            string_empty_data,
            [year_string, month_string, day_string, separator],
            "binary_join_element_wise",
            (),
        )

        # Only use nanosecond precision if necessary (nanoseconds provided).
        # This helps avoid overflow in some cases, and can allow us to represent more years.
        use_nanosecond_precision = nanosecond_expr is not None
        precision_str = "ns" if use_nanosecond_precision else "s"
        timestamp_empty_data = pd.Series(
            dtype=pd.ArrowDtype(pa.timestamp(precision_str))
        )
        duration_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.duration(precision_str)))

        # Pass date string to strptime to construct a timestamp from the components
        timestamp_expr = ArrowScalarFuncExpression(
            timestamp_empty_data,
            [date_string],
            "strptime",
            ("%Y-%m-%d", precision_str),
        )

        # We passed 1 as the day, so now we need to add the remaining time to the timestamp.
        # Convert days and time to nanosecond duration if needed or second duration

        # Subtract one from the day since we already have a timestamp on the first day of the month
        day_minus_one = ArithOpExpression(int_empty_data, day_expr, one_expr, "__sub__")
        day_scale_factor = ConstantExpression(
            int_empty_data,
            input_plan,
            86400 * (1_000_000_000 if use_nanosecond_precision else 1),
        )
        day_scaled = ArithOpExpression(
            int_empty_data, day_minus_one, day_scale_factor, "__mul__"
        )

        hour_scaled, minute_scaled, second_scaled = None, None, None
        if hour_expr:
            hour_scale_factor = ConstantExpression(
                int_empty_data,
                input_plan,
                3600 * (1_000_000_000 if use_nanosecond_precision else 1),
            )
            hour_scaled = ArithOpExpression(
                int_empty_data, hour_expr, hour_scale_factor, "__mul__"
            )
        if minute_expr:
            minute_scale_factor = ConstantExpression(
                int_empty_data,
                input_plan,
                60 * (1_000_000_000 if use_nanosecond_precision else 1),
            )
            minute_scaled = ArithOpExpression(
                int_empty_data, minute_expr, minute_scale_factor, "__mul__"
            )
        if second_expr and use_nanosecond_precision:
            second_scale_factor = ConstantExpression(
                int_empty_data, input_plan, 1_000_000_000
            )
            second_scaled = ArithOpExpression(
                int_empty_data, second_expr, second_scale_factor, "__mul__"
            )
        else:
            second_scaled = second_expr

        # Add up the nanoseconds/seconds from each day/time component
        additional_time = day_scaled
        for time_component_scaled in [
            hour_scaled,
            minute_scaled,
            second_scaled,
            nanosecond_expr,
        ]:
            if time_component_scaled:
                additional_time = ArithOpExpression(
                    int_empty_data,
                    additional_time,
                    time_component_scaled,
                    "__add__",
                )

        # Convert integer time to duration so it can be added to the timestamp.
        # CastExpressions are currently broken if the unit is not nanoseconds,
        # so in the seconds case we workaround by multiplying by a unit duration[s].
        if use_nanosecond_precision:
            additional_time_duration = CastExpression(
                duration_empty_data, additional_time
            )
        else:
            unit_duration = ConstantExpression(duration_empty_data, input_plan, 1)
            additional_time_duration = ArithOpExpression(
                duration_empty_data, additional_time, unit_duration, "__mul__"
            )

        # Add remaining time to timestamp
        timestamp_expr = ArithOpExpression(
            timestamp_empty_data,
            timestamp_expr,
            additional_time_duration,
            "__add__",
        )
        return timestamp_expr

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
            if func_name in ("DAYOFWEEK", "DOW"):
                # Default DAYOFWEEK for Bodo/Snowflake is Sunday=0, Monday=1, ..., Saturday=6.
                raw_expr = ArrowScalarFuncExpression(
                    empty_data, [input], arrow_func, (True, 7)
                )
            else:
                raw_expr = ArrowScalarFuncExpression(
                    empty_data, [input], arrow_func, ()
                )
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

        if func_name == "STR_TO_DATE" and num_operands == 2:
            input = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            mysql_fmt = str(java_call.getOperands()[1].toString())
            if mysql_fmt.startswith("'") and mysql_fmt.endswith("'"):
                mysql_fmt = mysql_fmt[1:-1]
            arrow_fmt = _mysql_date_format_to_arrow_format(mysql_fmt)

            timestamp_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.timestamp("ns")))
            timestamp_expr = ArrowScalarFuncExpression(
                timestamp_empty_data, [input], "strptime", (arrow_fmt, "ns")
            )

            # Cast to date at the end no matter what.
            # We need to truncate timestamps to be consistent with the JIT implementation.
            date_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.date32()))
            return CastExpression(date_empty_data, timestamp_expr)

        if func_name == "TIMESTAMP_FROM_PARTS":
            # Redirect to TZ-aware or TZ-naive version depending on whether a timezone is provided
            if num_operands == 8:
                func_name = "TIMESTAMP_TZ_FROM_PARTS"
            elif num_operands in (2, 6, 7):
                func_name = "TIMESTAMP_NTZ_FROM_PARTS"

        if func_name == "TIMESTAMP_NTZ_FROM_PARTS" and num_operands in (2, 6, 7):
            if num_operands in (6, 7):
                # Parts provided individually, with or without nanoseconds
                op_exprs = [
                    java_expr_to_python_expr(ctx, o, input_plan)
                    for o in java_call.getOperands()
                ]
                return timestamp_from_parts(*op_exprs)
            elif num_operands == 2:
                # Date expression and time expression provided
                date_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[0], input_plan
                )
                time_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[1], input_plan
                )

                timestamp_empty_data = pd.Series(
                    dtype=pd.ArrowDtype(pa.timestamp("ns"))
                )
                date_expr = CastExpression(timestamp_empty_data, date_expr)
                # time_expr can be a timestamp, in which case we need to cast to time64 to extract the time part
                time_expr = CastExpression(
                    pd.Series(dtype=pd.ArrowDtype(pa.time64("ns"))), time_expr
                )

                # Convert time to duration so it can be added to the date
                time_expr = CastExpression(
                    pd.Series(dtype=pd.ArrowDtype(pa.int64())), time_expr
                )
                time_expr = CastExpression(
                    pd.Series(dtype=pd.ArrowDtype(pa.duration("ns"))), time_expr
                )

                return ArithOpExpression(
                    timestamp_empty_data, date_expr, time_expr, "__add__"
                )

        if func_name == "TIMESTAMP_TZ_FROM_PARTS" and num_operands in (6, 7, 8):
            op_exprs = [
                java_expr_to_python_expr(ctx, o, input_plan)
                for o in java_call.getOperands()
            ]
            timestamp_expr = timestamp_from_parts(*op_exprs[:7])

            if num_operands == 8:
                timezone_expr = op_exprs[7]
                ensure_arg_is_const_expr_of_type(timezone_expr, "timezone_expr", str)
                timezone = timezone_expr.value
            else:
                timezone = ctx.default_tz if ctx.default_tz is not None else "UTC"

            timestamp_empty_data = pd.Series(
                dtype=pd.ArrowDtype(
                    pa.timestamp("ns" if num_operands > 6 else "s", tz=timezone)
                )
            )
            return ArrowScalarFuncExpression(
                timestamp_empty_data, [timestamp_expr], "assume_timezone", (timezone,)
            )

        if func_name == "TIMESTAMP_LTZ_FROM_PARTS" and num_operands in (6, 7):
            op_exprs = [
                java_expr_to_python_expr(ctx, o, input_plan)
                for o in java_call.getOperands()
            ]
            timestamp_expr = timestamp_from_parts(*op_exprs)
            local_tz = ctx.default_tz if ctx.default_tz is not None else "UTC"
            timestamp_empty_data = pd.Series(
                dtype=pd.ArrowDtype(
                    pa.timestamp("ns" if num_operands == 7 else "s", tz=local_tz)
                )
            )
            return ArrowScalarFuncExpression(
                timestamp_empty_data, [timestamp_expr], "assume_timezone", (local_tz,)
            )

        if func_name == "DATE_FROM_PARTS" and num_operands == 3:
            year_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            month_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[1], input_plan
            )
            day_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[2], input_plan
            )

            constructed_timestamp = timestamp_from_parts(
                year_expr, month_expr, day_expr
            )
            return CastExpression(
                pd.Series(dtype=pd.ArrowDtype(pa.date32())), constructed_timestamp
            )

        if func_name == "TIME_FROM_PARTS" and num_operands in (3, 4):
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))

            # If any of the inputs are floats, we will round to the nearest integer.
            # This is what the JIT implementation seems to do, even though float input isn't explicitly allowed in Snowflake.
            # See `time_from_parts` in BodoSQL/bodosql/kernels/time_array_kernels.py.
            def int_part_expr(op_index, part_expr_name):
                part_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[op_index], input_plan
                )
                ensure_type_of_expr(part_expr, part_expr_name, (int, float))
                part_expr_dtype = get_expr_dtype(part_expr, part_expr_name)
                if compare_types(part_expr_dtype, float):
                    part_expr = ArrowScalarFuncExpression(
                        pd.Series(dtype=pd.ArrowDtype(pa.float64())),
                        [part_expr],
                        "round",
                        (0, "half_towards_infinity"),
                    )
                    part_expr = CastExpression(int_empty_data, part_expr)
                return part_expr

            hour_expr = int_part_expr(0, "hour_expr")
            minute_expr = int_part_expr(1, "minute_expr")
            second_expr = int_part_expr(2, "second_expr")
            # Ensure seconds are int64 so the result of the later multiplication doesn't
            # end up being int32 internally, leading to overflow and the wrong answer.
            second_expr = CastExpression(int_empty_data, second_expr)

            # Get the nanoseconds for each part
            nanoseconds_per_hour = ConstantExpression(
                int_empty_data, input_plan, 3_600_000_000_000
            )
            hour_nanos = ArithOpExpression(
                int_empty_data, hour_expr, nanoseconds_per_hour, "__mul__"
            )
            nanoseconds_per_minute = ConstantExpression(
                int_empty_data, input_plan, 60_000_000_000
            )
            minute_nanos = ArithOpExpression(
                int_empty_data, minute_expr, nanoseconds_per_minute, "__mul__"
            )
            nanoseconds_per_second = ConstantExpression(
                int_empty_data, input_plan, 1_000_000_000
            )
            second_nanos = ArithOpExpression(
                int_empty_data, second_expr, nanoseconds_per_second, "__mul__"
            )

            total_nanos = ArithOpExpression(
                int_empty_data, hour_nanos, minute_nanos, "__add__"
            )
            total_nanos = ArithOpExpression(
                int_empty_data, total_nanos, second_nanos, "__add__"
            )

            if num_operands == 4:
                nanosecond_expr = int_part_expr(3, "nanosecond_expr")
                total_nanos = ArithOpExpression(
                    int_empty_data, total_nanos, nanosecond_expr, "__add__"
                )

            # If the total nanoseconds are negative or exceed 24 hours in nanoseconds, casting to
            # time64 will give us the nanoseconds modulo 24 hours, which is what we want.
            time_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.time64("ns")))
            return CastExpression(time_empty_data, total_nanos)

        if func_name == "MAKEDATE" and num_operands == 2:
            # MAKEDATE(year, dayofyear) → Jan 1 of year + (doy-1) days
            year_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            doy_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[1], input_plan
            )

            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            one_expr = ConstantExpression(int_empty_data, input_plan, 1)

            # DATE_FROM_PARTS accepts negative days whereas MAKEDATE doesn't,
            # so first check whether dayofyear is zero or negative.
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            invalid_doy = ComparisonOpExpression(
                bool_empty_data, doy_expr, one_expr, operator.lt
            )

            # Convert MAKEDATE(year, dayofyear) to DATE_FROM_PARTS(year, 1, dayofyear).
            result_timestamp = timestamp_from_parts(year_expr, one_expr, doy_expr)

            # If dayofyear is less than one, return NULL, else return the result date
            date_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.date32()))
            return CaseExpression(
                date_empty_data,
                invalid_doy,
                NullExpression(date_empty_data, input_plan, 0),
                result_timestamp,
            )

        if func_name == "DATE_TRUNC" and num_operands == 2:
            # DATE_TRUNC(FLAG(DAY), timestamp) → floor_temporal(timestamp, unit)
            unit_raw = get_java_symbol(java_call.getOperands()[0])
            unit_raw = standardize_java_time_unit(func_name, unit_raw)
            _TRUNC_UNITS = {
                "year",
                "quarter",
                "month",
                "week",
                "day",
                "hour",
                "minute",
                "second",
                "millisecond",
                "microsecond",
                "nanosecond",
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
                unit_str = get_java_symbol(java_call.getOperands()[1])
                unit_str = standardize_java_time_unit(func_name, unit_str).upper()
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

        if func_name in ("NEXT_DAY", "PREVIOUS_DAY") and num_operands == 2:
            date_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            dow_string_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[1], input_plan
            )
            ensure_type_of_expr(dow_string_expr, "dow_string_expr", str)

            # Arrow and DuckDB don't support timezone-aware compute so we fall back
            # to our JIT kernel.
            pa_type = date_expr.empty_data.iloc[:, 0].dtype.pyarrow_dtype
            if pa.types.is_timestamp(pa_type) and pa_type.tz is not None:
                return PythonScalarFuncExpression(
                    pd.Series(dtype=pd.ArrowDtype(pa.date32())),
                    [date_expr, dow_string_expr],
                    (
                        f"bodosql.kernels.datetime_array_kernels.{func_name.lower()}_wrapper",
                        False,  # is_series
                        False,  # is_method
                        (),  # args
                        {},  # kwargs
                        True,  # use_arrow_dtypes
                    ),
                    False,  # is_cfunc
                    False,  # has_state
                )

            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))

            # 0-6 integer matching order of Arrow's day_of_week
            target_dow_expr = ArrowScalarFuncExpression(
                pd.Series(dtype=pd.ArrowDtype(pa.int32())),
                [dow_string_expr],
                "day_of_week_num",
                (),
            )
            # Returns day of week between 0 and 6
            cur_dow_expr = ArrowScalarFuncExpression(
                int_empty_data, [date_expr], "day_of_week", ()
            )

            if func_name == "NEXT_DAY":
                # ((target_dow - cur_dow - 1) + 7) % 7 + 1 to get a day offset between 1 and 7
                diff_expr = ArithOpExpression(
                    int_empty_data, target_dow_expr, cur_dow_expr, "__sub__"
                )
            else:
                # ((cur_dow - target_dow - 1) + 7) % 7 + 1 to get a day offset between 1 and 7
                diff_expr = ArithOpExpression(
                    int_empty_data, cur_dow_expr, target_dow_expr, "__sub__"
                )
            diff_expr = ArithOpExpression(
                int_empty_data,
                diff_expr,
                ConstantExpression(int_empty_data, input_plan, 6),
                "__add__",
            )
            diff_mod_expr = ArithOpExpression(
                int_empty_data,
                diff_expr,
                ConstantExpression(int_empty_data, input_plan, 7),
                "__mod__",
            )
            days_to_offset_expr = ArithOpExpression(
                int_empty_data,
                diff_mod_expr,
                ConstantExpression(int_empty_data, input_plan, 1),
                "__add__",
            )

            # Get the number of seconds to add to or subtract from the input timestamp.
            # Arrow's duration type is the easiest way to add/subtract time, but it only accepts seconds and smaller units.
            seconds_in_day = ConstantExpression(int_empty_data, input_plan, 86400)
            seconds_to_offset_expr = ArithOpExpression(
                int_empty_data, days_to_offset_expr, seconds_in_day, "__mul__"
            )
            duration_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.duration("s")))
            one_duration_expr = ConstantExpression(duration_empty_data, input_plan, 1)
            seconds_to_offset_duration_expr = ArithOpExpression(
                duration_empty_data,
                seconds_to_offset_expr,
                one_duration_expr,
                "__mul__",
            )

            date_pa_type = date_expr.empty_data.iloc[:, 0].dtype.pyarrow_dtype
            if pa.types.is_date(date_pa_type):
                timestamp_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.timestamp("s")))
            else:
                # Otherwise, must be a timestamp
                timestamp_empty_data = date_expr.empty_data

            # Add or subtract the seconds offset to get the requested day
            next_day_timestamp_expr = ArithOpExpression(
                timestamp_empty_data,
                date_expr,
                seconds_to_offset_duration_expr,
                "__add__" if func_name == "NEXT_DAY" else "__sub__",
            )
            # According to Snowflake, we should always cast result to a date
            date_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.date32()))
            return CastExpression(date_empty_data, next_day_timestamp_expr)

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
                # and an interval literal.
                date_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[0], input_plan
                )
                amount_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[1], input_plan
                )
                int_empty = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
                out_empty = pd.Series(dtype=pd.ArrowDtype(pa.timestamp("ns")))

                amount_expr_dtype = get_expr_dtype(amount_expr, "DATEADD amount_expr")
                if compare_types(amount_expr_dtype, (int, float)):
                    # For the scalar fallback, default to DAY-based units
                    # (86_400_000_000_000 nanos per day)
                    nano_scale_expr = ConstantExpression(
                        int_empty, input_plan, 86_400_000_000_000
                    )

                    return ArrowScalarFuncExpression(
                        out_empty,
                        [
                            date_expr,
                            amount_expr,
                            ConstantExpression(int_empty, input_plan, 0),
                            nano_scale_expr,
                        ],
                        "bodo_dateadd",
                        (),
                    )
                else:
                    # Assume we have an interval type (e.g. a MonthDayNanoInterval tuple or duration type).
                    # Add the interval via ArithOpExpression, which should use DuckDB's calendar-aware
                    # interval arithmetic as long as amount_expr is a ConstantExpression.
                    return ArithOpExpression(
                        out_empty, date_expr, amount_expr, "__add__"
                    )
            elif num_operands == 3:
                # 3-operand form: DATEADD(unit, amount, date) → date + (unit * amount)
                # First operand is a FLAG(unit) interval qualifier from the Java
                # planner (e.g. FLAG(DAY), FLAG(MONTH)).
                unit_str = get_java_symbol(java_call.getOperands()[0])
                unit_str = standardize_java_time_unit(func_name, unit_str).upper()
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
        if func_name == "DATEDIFF" and num_operands == 3:
            # TODO: May need TZ-aware support

            # First operand is a FLAG(unit) interval qualifier from the Java
            # planner (e.g. FLAG(DAY), FLAG(MONTH)).
            unit_str = get_java_symbol(java_call.getOperands()[0])
            unit_str = standardize_java_time_unit(func_name, unit_str).upper()
            assert unit_str in INTERVAL_UNIT_MAP, (
                f"Unsupported DATEDIFF unit: {unit_str}"
            )
            # date_expr1 will be subtracted from date_expr2
            date_expr1 = java_expr_to_python_expr(
                ctx, java_call.getOperands()[1], input_plan
            )
            date_expr2 = java_expr_to_python_expr(
                ctx, java_call.getOperands()[2], input_plan
            )

            date1_pa_type = date_expr1.empty_data.iloc[:, 0].dtype.pyarrow_dtype
            date2_pa_type = date_expr2.empty_data.iloc[:, 0].dtype.pyarrow_dtype

            if (
                pa.types.is_timestamp(date1_pa_type) and date1_pa_type.tz is not None
            ) or (
                pa.types.is_timestamp(date2_pa_type) and date2_pa_type.tz is not None
            ):
                raise ValueError(
                    "TZ-aware input not currently supported in DATEDIFF (C++ backend)"
                )

            # Only mixing DATE and TIMESTAMP is allowed
            if pa.types.is_time(date1_pa_type) or pa.types.is_time(date2_pa_type):
                if unit_str in ("YEAR", "QUARTER", "MONTH", "WEEK", "DAY"):
                    raise ValueError(
                        "Unsupported unit for DATEDIFF with TIME input: " + unit_str
                    )
                if not (
                    pa.types.is_time(date1_pa_type) and pa.types.is_time(date2_pa_type)
                ):
                    raise ValueError(
                        "If a time type is provided both arguments must be time types."
                    )
            else:
                if pa.types.is_date(date1_pa_type):
                    if unit_str in (
                        "HOUR",
                        "MINUTE",
                        "SECOND",
                        "MILLISECOND",
                        "MICROSECOND",
                        "NANOSECOND",
                    ) or pa.types.is_timestamp(date2_pa_type):
                        timestamp_empty_data = pd.Series(
                            dtype=pd.ArrowDtype(pa.timestamp("ns"))
                        )
                        date_expr1 = CastExpression(timestamp_empty_data, date_expr1)
                if pa.types.is_date(date2_pa_type):
                    if unit_str in (
                        "HOUR",
                        "MINUTE",
                        "SECOND",
                        "MILLISECOND",
                        "MICROSECOND",
                        "NANOSECOND",
                    ) or pa.types.is_timestamp(date1_pa_type):
                        timestamp_empty_data = pd.Series(
                            dtype=pd.ArrowDtype(pa.timestamp("ns"))
                        )
                        date_expr2 = CastExpression(timestamp_empty_data, date_expr2)

            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))

            if unit_str == "YEAR":
                diff_func_name = "years_between"
            elif unit_str == "QUARTER":
                diff_func_name = "quarters_between"
            elif unit_str == "MONTH":
                diff_func_name = "month_interval_between"
                empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int32()))
            elif unit_str == "WEEK":
                # Depends on the week_start parameter
                diff_func_name = "weeks_between"
            elif unit_str == "DAY":
                diff_func_name = "days_between"
            elif unit_str == "HOUR":
                diff_func_name = "hours_between"
            elif unit_str == "MINUTE":
                diff_func_name = "minutes_between"
            elif unit_str == "SECOND":
                diff_func_name = "seconds_between"
            elif unit_str == "MILLISECOND":
                diff_func_name = "milliseconds_between"
            elif unit_str == "MICROSECOND":
                diff_func_name = "microseconds_between"
            elif unit_str == "NANOSECOND":
                diff_func_name = "nanoseconds_between"
            else:
                raise ValueError("DATEDIFF: Unrecognized unit " + unit_str)

            date_diff = ArrowScalarFuncExpression(
                empty_data, [date_expr1, date_expr2], diff_func_name, ()
            )
            return date_diff

        if func_name == "MONTHS_BETWEEN" and num_operands == 2:
            # date_expr2 will be subtracted from date_expr1
            date_expr1 = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            date_expr2 = java_expr_to_python_expr(
                ctx, java_call.getOperands()[1], input_plan
            )

            # date_expr1 > date_expr2 is the normal case yielding a positive result.
            # The math should still work out for date_expr1 < date_expr2 which equals -months_between(date_expr2, date_expr1).

            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            float_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            date_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.date32()))

            # Extract day parts of dates
            date1_day = ArrowScalarFuncExpression(
                int_empty_data, [date_expr1], "day", ()
            )
            date2_day = ArrowScalarFuncExpression(
                int_empty_data, [date_expr2], "day", ()
            )

            # Get the month interval between the dates.
            # This is just the number of crossed months, so it is not the correct output of MONTHS_BETWEEN.
            # We use month_interval_between to get the integer portion.
            # We adjust later if the day part of the later date is smaller than the day part of the earlier date.
            month_interval = ArrowScalarFuncExpression(
                pd.Series(dtype=pd.ArrowDtype(pa.int32())),
                [date_expr2, date_expr1],
                "month_interval_between",
                (),
            )
            month_interval = CastExpression(float_empty_data, month_interval)

            # Get the day component plus the time component of date1 and date2 in nanoseconds.
            # We do this by subtracting the full date from the date truncated to the month.
            date1_truncated = ArrowScalarFuncExpression(
                date_empty_data, [date_expr1], "floor_temporal", (1, "month")
            )
            date2_truncated = ArrowScalarFuncExpression(
                date_empty_data, [date_expr2], "floor_temporal", (1, "month")
            )
            date1_day_time = ArrowScalarFuncExpression(
                int_empty_data, [date1_truncated, date_expr1], "nanoseconds_between", ()
            )
            date2_day_time = ArrowScalarFuncExpression(
                int_empty_data, [date2_truncated, date_expr2], "nanoseconds_between", ()
            )

            # Get the nanosecond difference between the day and time components of the dates
            day_time_diff = ArithOpExpression(
                int_empty_data, date1_day_time, date2_day_time, "__sub__"
            )
            # It is okay if the difference is negative.
            # In this case, month_interval_between would have counted a month that was not a full elapsed month between the dates.
            # Therefore, adding a negative fraction to the integer month interval would just take away that extra whole month,
            # which is what we want.

            # Divide by 31 days to get the fraction of a month
            # 31 days = 2_678_000_000_000_000 nanoseconds
            month_nanos = ConstantExpression(
                int_empty_data, input_plan, 2_678_000_000_000_000
            )
            month_fraction = ArithOpExpression(
                float_empty_data, day_time_diff, month_nanos, "__truediv__"
            )

            # Add the fraction to the month interval (integer part) to get the complete result
            months_between = ArithOpExpression(
                float_empty_data, month_interval, month_fraction, "__add__"
            )

            # Special case: if the days are equal, the time portion is ignored and we can just return the month interval
            day_parts_equal = ComparisonOpExpression(
                bool_empty_data, date1_day, date2_day, operator.eq
            )

            # Special case: if both dates are on the last day of the month, we just return the integer month interval

            one_month_interval = ConstantExpression(
                date_empty_data,
                input_plan,
                ("MonthDayNanoInterval", 1, 0, 0),
            )
            one_day_interval = ConstantExpression(
                date_empty_data,
                input_plan,
                ("MonthDayNanoInterval", 0, 1, 0),
            )

            def get_last_day_of_month(date_truncated):
                next_month = ArithOpExpression(
                    date_empty_data,
                    date_truncated,
                    one_month_interval,
                    "__add__",
                )
                last_day_date = ArithOpExpression(
                    date_empty_data,
                    next_month,
                    one_day_interval,
                    "__sub__",
                )
                last_day = ArrowScalarFuncExpression(
                    int_empty_data, [last_day_date], "day", ()
                )
                return last_day

            date1_last_day = get_last_day_of_month(date1_truncated)
            date2_last_day = get_last_day_of_month(date2_truncated)
            is_date1_on_last_day = ComparisonOpExpression(
                bool_empty_data, date1_day, date1_last_day, operator.eq
            )
            is_date2_on_last_day = ComparisonOpExpression(
                bool_empty_data, date2_day, date2_last_day, operator.eq
            )
            both_dates_on_last_day = ConjunctionOpExpression(
                bool_empty_data, is_date1_on_last_day, is_date2_on_last_day, "__and__"
            )

            # Is it a special case where we are supposed to use the raw month interval?
            use_raw_month_interval = ConjunctionOpExpression(
                bool_empty_data, day_parts_equal, both_dates_on_last_day, "__or__"
            )

            return CaseExpression(
                float_empty_data, use_raw_month_interval, month_interval, months_between
            )

        if func_name == "TIME_SLICE":
            # TIME_SLICE(date, interval) → floor_temporal(date, interval)
            date_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            interval_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[1], input_plan
            )
            unit_str = get_java_symbol(java_call.getOperands()[2]).upper()
            start_or_end = "START"
            if num_operands == 4:
                start_or_end = get_java_symbol(java_call.getOperands()[3]).upper()
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

        if func_name == "YEARWEEK" and num_operands == 1:
            date_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )

            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            year_part = ArrowScalarFuncExpression(
                int_empty_data, [date_expr], "iso_year", ()
            )
            week_part = ArrowScalarFuncExpression(
                int_empty_data, [date_expr], "iso_week", ()
            )

            # Concatenate the year and the week parts in integer form
            year_part_shifted = ArithOpExpression(
                int_empty_data,
                year_part,
                ConstantExpression(int_empty_data, input_plan, 100),
                "__mul__",
            )
            week_part_appended = ArithOpExpression(
                int_empty_data, year_part_shifted, week_part, "__add__"
            )
            return week_part_appended

        if func_name in ("TIMEZONE_HOUR", "TIMEZONE_MINUTE") and num_operands == 1:
            timestamp_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )

            # Subtract the timestamp converted to UTC from the original timestamp
            # to get the nanoseconds between them.

            utc_timestamp_empty_data = pd.Series(
                dtype=pd.ArrowDtype(pa.timestamp("ns", tz="UTC"))
            )
            utc_timestamp_expr = CastExpression(
                utc_timestamp_empty_data, timestamp_expr
            )

            timestamp_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.timestamp("ns")))
            local_timestamp = ArrowScalarFuncExpression(
                timestamp_empty_data, [timestamp_expr], "local_timestamp", ()
            )
            local_utc_timestamp = ArrowScalarFuncExpression(
                timestamp_empty_data, [utc_timestamp_expr], "local_timestamp", ()
            )

            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            utc_offset_nanos = ArrowScalarFuncExpression(
                int_empty_data,
                [local_utc_timestamp, local_timestamp],
                "nanoseconds_between",
                (),
            )

            float_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            if func_name == "TIMEZONE_HOUR":
                # Get the hour part of the UTC offset using floor division
                nanoseconds_per_hour = ConstantExpression(
                    int_empty_data, input_plan, 3_600_000_000_000
                )
                return ArithOpExpression(
                    int_empty_data,
                    utc_offset_nanos,
                    nanoseconds_per_hour,
                    "__floordiv__",
                )
            else:
                # Get the minute part of the UTC offset using a modulo operation
                nanoseconds_per_minute = ConstantExpression(
                    int_empty_data, input_plan, 60_000_000_000
                )
                utc_offset_minutes = ArithOpExpression(
                    float_empty_data,
                    utc_offset_nanos,
                    nanoseconds_per_minute,
                    "__truediv__",
                )
                minutes_per_hour = ConstantExpression(int_empty_data, input_plan, 60)
                return ArithOpExpression(
                    float_empty_data, utc_offset_minutes, minutes_per_hour, "__mod__"
                )

        if func_name in (
            "EPOCH_SECOND",
            "EPOCH_MILLISECOND",
            "EPOCH_MICROSECOND",
            "EPOCH_NANOSECOND",
        ):
            timestamp_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )

            # Get the scale factor from the current timestamp unit to
            # the requested timestamp unit. This way we avoid a cast
            # to timestamp[ns] and eliminate any chance of overflow
            # during intermediate computation.
            def calculate_scale_factor(current_unit, target_unit):
                unit_scale = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}
                if unit_scale[target_unit] >= unit_scale[current_unit]:
                    return "__mul__", unit_scale[target_unit] / unit_scale[current_unit]
                else:
                    return "__floordiv__", unit_scale[current_unit] / unit_scale[
                        target_unit
                    ]

            if func_name == "EPOCH_SECOND":
                target_unit = "s"
            elif func_name == "EPOCH_MILLISECOND":
                target_unit = "ms"
            elif func_name == "EPOCH_MICROSECOND":
                target_unit = "us"
            elif func_name == "EPOCH_NANOSECOND":
                target_unit = "ns"

            timestamp_pa_type = timestamp_expr.empty_data.iloc[:, 0].dtype.pyarrow_dtype
            current_unit = timestamp_pa_type.unit

            # Calculate the scale factor and direction to go from the original
            # timestamp unit to the requested timestamp unit
            scale_op, scale_factor = calculate_scale_factor(current_unit, target_unit)

            # Convert to int to get epoch time (in the original timestamp units),
            # since Arrow timestamps are stored relative to the start of the UNIX epoch.
            # We don't need to cast to UTC to handle TZ-aware timestamps since
            # the underlying timestamp integer is always in UTC.
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            epoch_time = CastExpression(int_empty_data, timestamp_expr)

            if scale_factor > 1:
                scale_factor_expr = ConstantExpression(
                    int_empty_data, input_plan, scale_factor
                )
                return ArithOpExpression(
                    int_empty_data, epoch_time, scale_factor_expr, scale_op
                )
            else:
                return epoch_time

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

        if operand_type.getSqlTypeName().equals(
            SqlTypeName.VARCHAR
        ) and target_type.getSqlTypeName().equals(SqlTypeName.DATE):
            string_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.string()))

            # Replace any slashes with dashes since this is the format Arrow expects by default
            cleaned_input = ArrowScalarFuncExpression(
                string_empty_data, [in_expr], "replace_substring", ("/", "-")
            )

            # Add zeroes before the month and day if they are single digits
            cleaned_input = ArrowScalarFuncExpression(
                string_empty_data,
                [cleaned_input],
                "replace_substring_regex",
                (r"-(\d)-", r"-0\1-"),
            )
            cleaned_input = ArrowScalarFuncExpression(
                string_empty_data,
                [cleaned_input],
                "replace_substring_regex",
                (r"-(\d)($|[ T])", r"-0\1\2"),
            )

            # Cast to a timestamp first so we can parse string timestamps
            # before truncating to a date at the end.
            in_expr = CastExpression(
                pd.Series(dtype=pd.ArrowDtype(pa.timestamp("ns"))),
                cleaned_input,
            )

        if operand_type.getSqlTypeName().equals(
            SqlTypeName.VARCHAR
        ) and target_type.getSqlTypeName().equals(SqlTypeName.TIME):
            string_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.string()))
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))

            represents_int = ArrowScalarFuncExpression(
                pd.Series(dtype=pd.ArrowDtype(pa.bool_())),
                [in_expr],
                "utf8_is_digit",
                (),
            )

            # If the input represents simply an integer, interpret it as the number of seconds.
            # For the strings that were times, replace them with 0 so that we don't get an error trying to cast them to integers
            safe_int_strings = CaseExpression(
                string_empty_data,
                represents_int,
                in_expr,
                ConstantExpression(string_empty_data, input_plan, "0"),
            )
            seconds = CastExpression(int_empty_data, safe_int_strings)
            nanoseconds = ArithOpExpression(
                int_empty_data,
                seconds,
                ConstantExpression(int_empty_data, input_plan, 1_000_000_000),
                "__mul__",
            )
            nanoseconds_time = CastExpression(empty_data, nanoseconds)

            # Otherwise, we proceed with the usual parsing if the input string is formatted as a time

            # Add zeroes before the hour, minute, and second if they are single digits
            cleaned_input = ArrowScalarFuncExpression(
                string_empty_data,
                [in_expr],
                "replace_substring_regex",
                (r"^(\d):", r"0\1:"),
            )
            cleaned_input = ArrowScalarFuncExpression(
                string_empty_data,
                [cleaned_input],
                "replace_substring_regex",
                (r":(\d):", r":0\1:"),
            )
            cleaned_input = ArrowScalarFuncExpression(
                string_empty_data,
                [cleaned_input],
                "replace_substring_regex",
                (r":(\d)($|\.)", r":0\1\2"),
            )

            # For the strings that were integers, replace them with 00:00:00 so that we don't get an error trying to parse integers as times
            safe_time_strings = CaseExpression(
                string_empty_data,
                represents_int,
                ConstantExpression(string_empty_data, input_plan, "00:00:00"),
                cleaned_input,
            )

            # Convert to a timestamp string before casting because Arrow has no way to directly parse as a time.
            dummy_date_string = ConstantExpression(
                string_empty_data, input_plan, "1970-01-01"
            )
            separator = ConstantExpression(string_empty_data, input_plan, " ")
            timestamp_string = ArrowScalarFuncExpression(
                string_empty_data,
                [dummy_date_string, safe_time_strings, separator],
                "binary_join_element_wise",
                (),
            )
            # Parse as timestamp
            timestamp_expr = CastExpression(
                pd.Series(dtype=pd.ArrowDtype(pa.timestamp("ns"))), timestamp_string
            )
            # Casting to time64 will strip off the dummy date part of the timestamp
            timestamp_time = CastExpression(empty_data, timestamp_expr)

            # Return the final time64 array, selecting the result based on whether each input was an integer string or time string
            return CaseExpression(
                empty_data, represents_int, nanoseconds_time, timestamp_time
            )

        # TO_TIMESTAMP/TO_TIMESTAMP_NTZ remove the timezone which is same as
        # local_timestamp() function of Arrow (not cast)
        if (
            operand_type.getSqlTypeName().equals(
                SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE
            )
            or operand_type.getSqlTypeName().equals(SqlTypeName.TIMESTAMP_TZ)
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

        # Integers are assumed in seconds in BodoSQL (instead of nanoseconds as converted by sql_type_to_pa_type())
        if is_int_type(operand_type) and target_type.getSqlTypeName().equals(
            SqlTypeName.TIMESTAMP
        ):
            # Arrow's cast_timestamp only accepts int64 input, so convert other integer types to int64 first.
            # This likely won't work for uint64 where the input is greater than the max value of int64,
            # but the timestamp itself is backed by signed int64 anyway
            in_expr_dtype = get_expr_dtype(in_expr)
            if not compare_types(in_expr_dtype, "int64"):
                int64_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
                in_expr = CastExpression(
                    int64_empty_data,
                    in_expr,
                )

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

        if kind.equals(SqlKind.IS_NOT_TRUE):
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return UnaryOpExpression(bool_empty_data, input, "isnottrue")

        if kind.equals(SqlKind.IS_FALSE):
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return UnaryOpExpression(bool_empty_data, input, "isfalse")

        if kind.equals(SqlKind.IS_NOT_FALSE):
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return UnaryOpExpression(bool_empty_data, input, "isnotfalse")

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
        unit_str = get_java_symbol(java_call.getOperands()[0]).upper()
        input = java_expr_to_python_expr(ctx, java_call.getOperands()[1], input_plan)
        arrow_func = _DATE_PART_ARROW_FUNCS.get(unit_str)
        if arrow_func is None:
            raise NotImplementedError(f"Unsupported EXTRACT unit: {unit_str}")
        empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))

        if unit_str in ("DAYOFWEEK", "DOW"):
            # Default DAYOFWEEK for Bodo/Snowflake is Sunday=0, Monday=1, ..., Saturday=6.
            raw_expr = ArrowScalarFuncExpression(
                empty_data, [input], arrow_func, (True, 7)
            )
        else:
            raw_expr = ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())
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
            "WEEKDAY",
            "WEEK",
            "WEEKOFYEAR",
            "WEEKISO",
            "YEAROFWEEK",
            "YEAROFWEEKISO",
            "HOUR",
            "MINUTE",
            "SECOND",
            "QUARTER",
            "MICROSECOND",
            "NANOSECOND",
        ):
            # TODO: Properly differentiate between the ISO and non-ISO versions.
            # Currently we resort to Arrow's ISO versions for the variants aside from regular DAYOFWEEK.
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            return ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())

        if func_name == "DAYOFWEEKISO":
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            # Set count_from_zero=False and week_start=1 which corresponds to Monday.
            # Therefore we get Monday=1, Tuesday=2, ..., Sunday=7.
            return ArrowScalarFuncExpression(
                empty_data, [input], arrow_func, (False, 1)
            )

        if func_name in ("DAYOFWEEK", "DOW"):
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            # Pass week_start = 7 which corresponds to Sunday, so we have Sunday=0, Monday=1, ..., Saturday=6.
            return ArrowScalarFuncExpression(empty_data, [input], arrow_func, (True, 7))

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
            tz = ctx.default_tz if ctx.default_tz is not None else "UTC"
            curr_ts = pd.Timestamp.now(tz=tz)
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

        def cur_local_timestamp():
            tz = ctx.default_tz if ctx.default_tz is not None else "UTC"
            curr_ts = pd.Timestamp.now(tz=tz)
            dummy_empty_data = pd.Series(
                [curr_ts], dtype=pd.ArrowDtype(pa.timestamp("ns", tz=tz))
            )
            return ConstantExpression(dummy_empty_data, input_plan, curr_ts)

        if func_name in (
            "CURRENT_TIMESTAMP",
            "GETDATE",
            "LOCALTIMESTAMP",
            "SYSTIMESTAMP",
            "NOW",
        ):
            return cur_local_timestamp()

        if func_name == "UNIX_TIMESTAMP":
            # This is like EPOCH_SECOND but we retain any fractional seconds.
            # If 0 arguments are provided, we have to get the current timestamp
            # before calculating the seconds.

            if len(op_exprs) == 0:
                # The timezone of the timestamp here doesn't matter as long
                # as it is TZ-aware.
                timestamp_expr = cur_local_timestamp()
            elif len(op_exprs) == 1:
                timestamp_expr = op_exprs[0]
                # If input is not TZ-aware, interpret as in local time
                timestamp_pa_type = timestamp_expr.empty_data.iloc[
                    :, 0
                ].dtype.pyarrow_dtype
                if pa.types.is_date(timestamp_pa_type) or (
                    pa.types.is_timestamp(timestamp_pa_type)
                    and timestamp_pa_type.tz is None
                ):
                    tz = ctx.default_tz if ctx.default_tz is not None else "UTC"
                    local_timestamp_empty_data = pd.Series(
                        dtype=pd.ArrowDtype(pa.timestamp("ns", tz=tz))
                    )
                    timestamp_expr = ArrowScalarFuncExpression(
                        local_timestamp_empty_data,
                        [timestamp_expr],
                        "assume_timezone",
                        (tz,),
                    )

            # Convert to int to get epoch nanoseconds, since Arrow timestamps are
            # stored in UTC relative to the start of the UNIX epoch
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            timestamp_nanos = CastExpression(int_empty_data, timestamp_expr)

            # To get seconds since epoch as float, we need to divide by 1 billion.
            # Using __truediv__ directly will fail because it will attempt to cast
            # the operands to float without losing precision, and timestamp_nanos
            # is simply too large.

            # To work around this we need to calculate the integer part and fractional
            # part separately, and then add them together.

            # Get integer seconds since epoch
            nanoseconds_per_second = ConstantExpression(
                int_empty_data, input_plan, 1_000_000_000
            )
            timestamp_seconds_int = ArithOpExpression(
                int_empty_data, timestamp_nanos, nanoseconds_per_second, "__floordiv__"
            )

            # Get fraction of a second unaccounted for
            timestamp_ns_remainder = ArithOpExpression(
                int_empty_data, timestamp_nanos, nanoseconds_per_second, "__mod__"
            )
            float_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            timestamp_sec_remainder = ArithOpExpression(
                float_empty_data,
                timestamp_ns_remainder,
                nanoseconds_per_second,
                "__truediv__",
            )

            timestamp_seconds = ArithOpExpression(
                float_empty_data,
                timestamp_seconds_int,
                timestamp_sec_remainder,
                "__add__",
            )

            # MySQL only accepts timestamps after 1970-01-01 00:00:01, else the function returns 0.
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            one_expr = ConstantExpression(float_empty_data, input_plan, 1.0)
            under_one_second = ComparisonOpExpression(
                bool_empty_data, timestamp_seconds, one_expr, operator.lt
            )
            zero_expr = ConstantExpression(float_empty_data, input_plan, 0.0)
            return CaseExpression(
                float_empty_data, under_one_second, zero_expr, timestamp_seconds
            )

        if func_name == "FROM_UNIXTIME" and len(op_exprs) == 1:
            epoch_seconds_expr = op_exprs[0]
            ensure_type_of_expr(epoch_seconds_expr, "epoch_seconds_expr", (int, float))

            # Return timestamp in local timezone
            tz = ctx.default_tz if ctx.default_tz is not None else "UTC"

            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))

            epoch_seconds_type = get_expr_dtype(epoch_seconds_expr)
            if compare_types(epoch_seconds_type, int):
                # We need this to be timestamp[s] so that the integer input in seconds
                # can be interpreted as in the proper unit when casting
                timestamp_s_empty_data = pd.Series(
                    dtype=pd.ArrowDtype(pa.timestamp("s", tz=tz))
                )

                # Ensure input is int64 so we can cast to timestamp type
                epoch_seconds_expr = CastExpression(int_empty_data, epoch_seconds_expr)
                # Convert seconds since epoch to timestamp
                return CastExpression(timestamp_s_empty_data, epoch_seconds_expr)
            else:
                # Input is a float, so we should use nanosecond resolution.
                timestamp_ns_empty_data = pd.Series(
                    dtype=pd.ArrowDtype(pa.timestamp("ns", tz=tz))
                )

                # Convert input to nanoseconds
                nanoseconds_per_second = ConstantExpression(
                    int_empty_data, input_plan, 1_000_000_000
                )
                float_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
                epoch_nanos = ArithOpExpression(
                    float_empty_data,
                    epoch_seconds_expr,
                    nanoseconds_per_second,
                    "__mul__",
                )

                # Need to round in case the input converted to nanoseconds
                # is still not an integer.
                epoch_nanos = UnaryOpExpression(int_empty_data, epoch_nanos, "round")

                # Convert nanoseconds since epoch to timestamp
                return CastExpression(timestamp_ns_empty_data, epoch_nanos)

        if func_name == "FROM_DAYS" and len(op_exprs) == 1:
            """Get date a number of days after January 1st of year 0"""
            days_expr = op_exprs[0]
            ensure_type_of_expr(days_expr, "days_expr", (int, float))

            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))

            # If provided days are a float, we should round to match MySQL
            days_expr_dtype = get_expr_dtype(days_expr)
            if compare_types(days_expr_dtype, float):
                days_expr = UnaryOpExpression(int_empty_data, days_expr, "round")

            # Convert to the number of days of the start of the UNIX epoch, since this is how date32 stores dates
            year_zero_to_epoch_start_days = ConstantExpression(
                int_empty_data, input_plan, 719528
            )
            days_after_epoch = ArithOpExpression(
                int_empty_data, days_expr, year_zero_to_epoch_start_days, "__sub__"
            )

            # Input needs to be int32 to cast to date32.
            # If the number of days didn't fit in int32, it would be far too large anyway.
            days_after_epoch = CastExpression(
                pd.Series(dtype=pd.ArrowDtype(pa.int32())), days_after_epoch
            )

            # Cast epoch days to date type
            date32_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.date32()))
            date_expr = CastExpression(date32_empty_data, days_after_epoch)

            # If days <= 365, MySQL returns 0000-00-00.
            # This is not possible for us, so we return NULL instead.
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            invalid_days = ComparisonOpExpression(
                bool_empty_data,
                days_expr,
                ConstantExpression(int_empty_data, input_plan, 365),
                operator.le,
            )
            null_date = NullExpression(date32_empty_data, input_plan, 0)
            return CaseExpression(date32_empty_data, invalid_days, null_date, date_expr)

        if func_name == "TO_DAYS" and len(op_exprs) == 1:
            """Get the number of days between January 1st of year 0
            and the input date"""
            date_expr = op_exprs[0]

            # date_expr could be a timestamp, so first truncate to a date.
            # If date_expr is TZ-aware, this will be the local timestamp date.
            date_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.date32()))
            date_expr = CastExpression(date_empty_data, date_expr)

            # Cast date32 to int32 to get days since start of epoch
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int32()))
            days_after_epoch = CastExpression(int_empty_data, date_expr)

            # Convert to days since year 0
            year_zero_to_epoch_start_days = ConstantExpression(
                int_empty_data, input_plan, 719528
            )
            return ArithOpExpression(
                int_empty_data,
                days_after_epoch,
                year_zero_to_epoch_start_days,
                "__add__",
            )

        if func_name == "TO_SECONDS" and len(op_exprs) == 1:
            """Get the number of seconds between January 1st of year 0
            and the input timestamp"""
            timestamp_expr = op_exprs[0]

            timestamp_expr_dtype = timestamp_expr.empty_data.iloc[
                :, 0
            ].dtype.pyarrow_dtype
            if pa.types.is_date(timestamp_expr_dtype):
                # If the input is a date, convert to a timestamp (in seconds)
                timestamp_unit = "s"
                timestamp_s_empty_data = pd.Series(
                    dtype=pd.ArrowDtype(pa.timestamp(timestamp_unit))
                )
                timestamp_expr = CastExpression(timestamp_s_empty_data, timestamp_expr)
            elif pa.types.is_timestamp(timestamp_expr_dtype):
                timestamp_unit = timestamp_expr_dtype.unit
                # If the timestamp is TZ-aware, use the local timestamp
                if timestamp_expr_dtype.tz is not None:
                    timestamp_empty_data = pd.Series(
                        dtype=pd.ArrowDtype(pa.timestamp(timestamp_unit))
                    )
                    timestamp_expr = ArrowScalarFuncExpression(
                        timestamp_expr.empty_data, timestamp_expr, "local_timestamp", ()
                    )
            else:
                raise ValueError("TO_SECONDS: Unsupported input type")

            # Cast timestamp to int64 to get the timestamp value in the original units
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            timestamp_int = CastExpression(int_empty_data, timestamp_expr)

            if timestamp_unit == "s":
                # No division needed
                timestamp_seconds = timestamp_int
            else:
                # Determine what to divide by get seconds
                if timestamp_unit == "ms":
                    divisor = 1_000
                elif timestamp_unit == "us":
                    divisor = 1_000_000
                elif timestamp_unit == "ns":
                    divisor = 1_000_000_000
                # Get seconds since start of epoch using floor division.
                # (MySQL appears not to round fractional seconds)
                divisor_expr = ConstantExpression(int_empty_data, input_plan, divisor)
                timestamp_seconds = ArithOpExpression(
                    int_empty_data, timestamp_int, divisor_expr, "__floordiv__"
                )

            # Adjust from seconds since epoch to seconds since year 0
            year_zero_to_epoch_start_seconds = ConstantExpression(
                int_empty_data, input_plan, 62167219200
            )
            return ArithOpExpression(
                int_empty_data,
                timestamp_seconds,
                year_zero_to_epoch_start_seconds,
                "__add__",
            )

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

        if func_name == "SIGN" and len(op_exprs) == 1:
            inp = op_exprs[0]
            ensure_type_of_expr(inp, func_name + " input", (int, float))

            inp_dtype = get_expr_dtype(inp, func_name + " input")
            if compare_types(inp_dtype, int):
                # If input is an int, first use int8 empty data since
                # we have to match the return type of Arrow's sign().
                int8_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int8()))
                int8_sign = UnaryOpExpression(int8_empty_data, inp, "sign")

                # Calcite retains the size of the input integer type.
                # Using inp.empty_data directly could be problematic
                # if it is unsigned, so cast to the equivalent pyarrow
                # type of what is in the plan.
                inp_sql_type = java_call.getOperands()[0].getType()
                pa_int_type = sql_type_to_pa_type(ctx, inp_sql_type.getSqlTypeName())
                sql_pa_empty = pd.Series(dtype=pd.ArrowDtype(pa_int_type))
                return CastExpression(sql_pa_empty, int8_sign)
            else:
                # If input is a float, return the original float type
                return UnaryOpExpression(inp.empty_data, inp, "sign")

        # Binary power: POWER(x, y) -> use __pow__ via ArithOpExpression
        if func_name == "POWER" and len(op_exprs) == 2:
            left = op_exprs[0]
            right = op_exprs[1]
            ensure_type_of_expr(left, func_name + " left input", (int, float))
            ensure_type_of_expr(right, func_name + " right input", (int, float))
            out_empty = left.empty_data.iloc[:, 0] ** right.empty_data.iloc[:, 0]
            return ArithOpExpression(out_empty, left, right, "__pow__")

        # SQRT(x) -> unary sqrt
        if func_name == "SQRT" and len(op_exprs) == 1:
            inp = op_exprs[0]
            ensure_type_of_expr(inp, func_name + " input", (int, float))
            out_empty = inp.empty_data.iloc[:, 0] ** 0.5
            return UnaryOpExpression(out_empty, inp, "sqrt")

        # CBRT(x) -> unary cube root
        if func_name == "CBRT" and len(op_exprs) == 1:
            inp = op_exprs[0]
            ensure_type_of_expr(inp, func_name + " input", (int, float))
            out_empty = inp.empty_data
            return UnaryOpExpression(out_empty, inp, "cbrt")

        # ABS(x)
        if func_name == "ABS" and len(op_exprs) == 1:
            inp = op_exprs[0]
            ensure_type_of_expr(inp, func_name + " input", (int, float))
            out_empty = inp.empty_data.iloc[:, 0].abs()
            return UnaryOpExpression(out_empty, inp, "abs")

        # CEILING(x)
        if func_name == "CEILING" and len(op_exprs) == 1:
            # Redirect to CEIL below.
            func_name = "CEIL"

        if func_name in ("FLOOR", "CEIL") and len(op_exprs) == 1:
            inp = op_exprs[0]
            ensure_type_of_expr(inp, func_name + " input", (int, float))

            inp_dtype = get_expr_dtype(inp, func_name + " input")
            if compare_types(inp_dtype, int):
                # If input is an integer, FLOOR/CEIL is a no-op
                return inp
            else:
                # If input is a float, return FLOOR(inp) or CEIL(inp) as normal
                return UnaryOpExpression(inp.empty_data, inp, func_name.lower())

        # EXP(x)
        if func_name == "EXP" and len(op_exprs) == 1:
            inp = op_exprs[0]
            ensure_type_of_expr(inp, "EXP input", (int, float))

            # Retain current float output type if input is a float,
            # otherwise use float64.
            inp_dtype = get_expr_dtype(inp, "EXP input")
            if compare_types(inp_dtype, int):
                out_empty = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            else:
                out_empty = inp.empty_data

            return UnaryOpExpression(out_empty, inp, "exp")

        # LN(x) -> natural log
        if func_name == "LN" and len(op_exprs) == 1:
            inp = op_exprs[0]
            ensure_type_of_expr(inp, "LN input", (int, float))

            # Retain current float output type if input is a float,
            # otherwise use float64.
            inp_dtype = get_expr_dtype(inp, "LN input")
            if compare_types(inp_dtype, int):
                out_empty = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            else:
                out_empty = inp.empty_data

            return UnaryOpExpression(out_empty, inp, "ln")

        elif func_name == "LOG10" and len(op_exprs) == 1:
            inp = op_exprs[0]
            ensure_type_of_expr(inp, "LOG10 input", (int, float))
            out_empty = inp.empty_data
            return UnaryOpExpression(out_empty, inp, "log10")

        # ROUND(x, d) or ROUND(x) -> map to a unary/binary op if supported
        if func_name == "ROUND" and len(op_exprs) in (1, 2):
            inp = op_exprs[0]
            ensure_type_of_expr(inp, "ROUND input", (int, float))
            out_empty = inp.empty_data

            if len(op_exprs) == 1:
                inp_dtype = get_expr_dtype(inp, "ROUND input")
                if compare_types(inp_dtype, int):
                    # If input is an integer, single-argument ROUND is a no-op
                    return inp
                else:
                    return UnaryOpExpression(out_empty, inp, "round")
            else:
                precision_digits = op_exprs[1]
                # Not a traditional arithmetic operation, but this is what
                # we currently have available to retrieve binary functions
                # from the DuckDB catalog.
                return ArithOpExpression(out_empty, inp, precision_digits, "round")

        if func_name == "TRUNCATE" and len(op_exprs) == 1:
            inp = op_exprs[0]
            ensure_type_of_expr(inp, func_name + " input", (int, float))

            inp_dtype = get_expr_dtype(inp, func_name + " input")
            if compare_types(inp_dtype, int):
                # If input is an integer, TRUNCATE is a no-op
                return inp
            else:
                # If input is a float, return trunc(inp) as normal
                return UnaryOpExpression(inp.empty_data, inp, "trunc")

        if func_name == "MOD" and len(op_exprs) == 2:
            inp = op_exprs[0]
            modulus_expr = op_exprs[1]
            ensure_type_of_expr(inp, "MOD inp", int)
            ensure_type_of_expr(modulus_expr, "modulus_expr", int)

            return ArithOpExpression(inp.empty_data, inp, modulus_expr, "__mod__")

        if func_name == "RAND" and len(op_exprs) == 0:
            """Generates random doubles in the range [0, 1)"""
            # Create a dummy expression from the input plan which will be used on the
            # C++ side to check the row count (number of random values to generate).
            row_count_info_expr = ConstantExpression(
                pd.Series(dtype=pd.ArrowDtype(pa.int64())), input_plan, 0
            )
            # Note that we pass "rand" instead of "random" to distinguish between Arrow's random()
            # and this, which take different arguments. (Though we do ultimately rely on random())
            return ArrowScalarFuncExpression(
                pd.Series(dtype=pd.ArrowDtype(pa.float64())),
                [row_count_info_expr],
                "rand",
                (),
            )

        if func_name in ("BOOLAND", "BOOLOR", "BOOLXOR") and len(op_exprs) == 2:
            left_expr = op_exprs[0]
            right_expr = op_exprs[1]

            ensure_type_of_expr(left_expr, "left_expr", (int, float))
            ensure_type_of_expr(right_expr, "right_expr", (int, float))

            left_expr_is_int = compare_types(get_expr_dtype(left_expr), int)
            right_expr_is_int = compare_types(get_expr_dtype(right_expr), int)

            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            float_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))

            if not left_expr_is_int:
                # Round float inputs.
                # This is required because, according to the docs,
                # Snowflake interprets floats in BOOLAND as integers
                # by rounding. Thus, a float like 0.3 should be considered 0
                # whereas a float like 0.7 would be non-zero.

                left_expr_rounded = ArrowScalarFuncExpression(
                    float_empty_data, [left_expr], "round", ()
                )
            else:
                left_expr_rounded = left_expr

            if not right_expr_is_int:
                right_expr_rounded = ArrowScalarFuncExpression(
                    float_empty_data, [right_expr], "round", ()
                )
            else:
                right_expr_rounded = right_expr

            # Get nonzero values as True, zero values as False
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            left_zero_expr = ConstantExpression(
                int_empty_data if left_expr_is_int else float_empty_data, input_plan, 0
            )
            left_expr_bool = ComparisonOpExpression(
                bool_empty_data, left_expr_rounded, left_zero_expr, operator.ne
            )
            right_zero_expr = ConstantExpression(
                int_empty_data if right_expr_is_int else float_empty_data, input_plan, 0
            )
            right_expr_bool = ComparisonOpExpression(
                bool_empty_data, right_expr_rounded, right_zero_expr, operator.ne
            )

            if func_name == "BOOLAND":
                return ConjunctionOpExpression(
                    bool_empty_data, left_expr_bool, right_expr_bool, "__and__"
                )
            elif func_name == "BOOLOR":
                return ConjunctionOpExpression(
                    bool_empty_data, left_expr_bool, right_expr_bool, "__or__"
                )
            elif func_name == "BOOLXOR":
                return ComparisonOpExpression(
                    bool_empty_data, left_expr_bool, right_expr_bool, operator.ne
                )

        if func_name == "BOOLNOT" and len(op_exprs) == 1:
            expr = op_exprs[0]
            ensure_type_of_expr(expr, "expr", (int, float))

            expr_is_int = compare_types(get_expr_dtype(expr), int)

            # Round float inputs (for the same reason as BOOLAND/BOOLOR/BOOLXOR)
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            float_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))

            if not expr_is_int:
                expr_rounded = ArrowScalarFuncExpression(
                    float_empty_data, [expr], "round", ()
                )
            else:
                expr_rounded = expr

            # Flipped logic: get nonzero values as False, zero values as True
            zero_expr = ConstantExpression(
                int_empty_data if expr_is_int else float_empty_data, input_plan, 0
            )
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return ComparisonOpExpression(
                bool_empty_data, expr_rounded, zero_expr, operator.eq
            )

        def equal_null(left_expr, right_expr):
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            values_equal = ComparisonOpExpression(
                bool_empty_data, left_expr, right_expr, operator.eq
            )

            left_is_null = UnaryOpExpression(bool_empty_data, left_expr, "isnull")
            right_is_null = UnaryOpExpression(bool_empty_data, right_expr, "isnull")
            both_null = ConjunctionOpExpression(
                bool_empty_data, left_is_null, right_is_null, "__and__"
            )

            # CASE WHEN values_equal THEN TRUE ELSE both_null
            # The case statement interprets nulls as false, so we avoid the coalesce step
            true_expr = ConstantExpression(bool_empty_data, input_plan, True)
            return CaseExpression(bool_empty_data, values_equal, true_expr, both_null)

        if func_name == "EQUAL_NULL" and len(op_exprs) == 2:
            return equal_null(op_exprs[0], op_exprs[1])

        if func_name == "IFF" and len(op_exprs) == 3:
            # IFF is equivalent to CASE with single WHEN
            return java_case_to_python_case(ctx, operands, input_plan)

        if func_name == "NULLIF" and len(op_exprs) == 2:
            return ArrowScalarFuncExpression(
                op_exprs[0].empty_data, op_exprs, "nullif", ()
            )

        if func_name == "NVL2" and len(op_exprs) == 3:
            expr1 = op_exprs[0]
            expr2 = op_exprs[1]
            expr3 = op_exprs[2]

            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            expr1_is_null = UnaryOpExpression(bool_empty_data, expr1, "isnull")
            return make_unified_case_expression("common", expr1_is_null, expr3, expr2)

        if func_name == "ZEROIFNULL" and len(op_exprs) == 1:
            expr = op_exprs[0]
            ensure_type_of_expr(expr, "ZEROIFNULL expr", (int, float))

            zero_expr = ConstantExpression(expr.empty_data, input_plan, 0)
            return ArrowScalarFuncExpression(
                expr.empty_data, [expr, zero_expr], "coalesce", ()
            )

        if func_name == "REGR_VALX" and len(op_exprs) == 2:
            y_expr = op_exprs[0]
            x_expr = op_exprs[1]

            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            y_is_null = UnaryOpExpression(bool_empty_data, y_expr, "isnull")
            return make_unified_case_expression(
                x_expr.empty_data, y_is_null, y_expr, x_expr
            )

        if func_name == "REGR_VALY" and len(op_exprs) == 2:
            y_expr = op_exprs[0]
            x_expr = op_exprs[1]

            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            x_is_null = UnaryOpExpression(bool_empty_data, x_expr, "isnull")
            return make_unified_case_expression(
                y_expr.empty_data, x_is_null, x_expr, y_expr
            )

        if func_name == "DECODE" and len(op_exprs) >= 3:
            select_expr = op_exprs[0]

            result_exprs = [op_exprs[i] for i in range(2, len(op_exprs), 2)]
            # Ensure result expression datatypes are compatible. Currently we
            # only try to unify integer datatypes.
            result_expr_dtypes = [
                get_expr_dtype(result_expr) for result_expr in result_exprs
            ]
            if not all(
                pd.api.types.is_integer_dtype(dtype) for dtype in result_expr_dtypes
            ):
                result_type = result_expr_dtypes[0]
                for result_expr_dtype in result_expr_dtypes[1:]:
                    if not compare_types(result_expr_dtype, result_type):
                        raise ValueError(
                            f"Incompatible DECODE result expression dtypes: {result_expr_dtype} and {result_type}"
                        )

            # Get unified result type between all result expressions to avoid overflow
            common_result_type, results_need_cast = get_common_int_type_list(
                result_exprs
            )
            if common_result_type is not None:
                empty_data = pd.Series(dtype=pd.ArrowDtype(common_result_type))
            else:
                empty_data = result_exprs[0].empty_data

            if len(op_exprs) % 2 == 0:
                # Default specified
                default_expr = op_exprs[-1]
            else:
                # No default specified - NULL should be returned when there is no match
                default_expr = NullExpression(empty_data, input_plan, 0)

            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))

            # Recursively create CaseExpressions for the remaining search expression and result expression pairs
            # There is guaranteed to be at least one pair
            def make_ternary_expression(search_result_pair_index):
                pair_start_index = 1 + 2 * search_result_pair_index
                search_expr = op_exprs[pair_start_index]
                result_expr = op_exprs[pair_start_index + 1]
                if results_need_cast[search_result_pair_index]:
                    result_expr = CastExpression(empty_data, result_expr)

                # Use equal_null for comparison since nulls should match nulls
                search_expr_match = equal_null(select_expr, search_expr)
                # Get the result expression if select_expr does not match search_expr
                if len(op_exprs) - pair_start_index > 3:
                    else_ternary_expr = make_ternary_expression(
                        search_result_pair_index + 1
                    )
                else:
                    else_ternary_expr = default_expr  # Use the default expression if we have run out of pairs
                return CaseExpression(
                    empty_data, search_expr_match, result_expr, else_ternary_expr
                )

            return make_ternary_expression(0)

        if func_name == "CHAR_LENGTH" and len(op_exprs) == 1:
            src = op_exprs[0]
            ensure_type_of_expr(src, "src", (str, pa.binary()))
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
            ensure_type_of_expr(src, "src", (str, pa.binary()))

            length = op_exprs[1]
            ensure_arg_is_const_expr_of_type(length, "length", int)

            arrow_func_args = (length.value,)

            if len(op_exprs) == 3:
                pattern = op_exprs[2]
                ensure_arg_is_const_expr_of_type(pattern, "pattern", (str, pa.binary()))
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

            ensure_type_of_expr(src, "src", (str, pa.binary()))
            ensure_arg_is_const_expr_of_type(
                search_expr, "search_expr", (str, pa.binary())
            )

            if len(op_exprs) == 3:
                replacement_expr = op_exprs[2]
                ensure_arg_is_const_expr_of_type(
                    replacement_expr, "replacement_expr", (str, pa.binary())
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

            def clean_regex_params(regex_params_expr):
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
                return regex_params

            src = op_exprs[0]
            regexp = op_exprs[1]

            ensure_type_of_expr(src, "src", (str, pa.binary()))
            ensure_arg_is_const_expr_of_type(regexp, "regexp", (str, pa.binary()))

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
                regex_params = clean_regex_params(regex_params_expr)
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
            if start > 0:
                without_start_expr = ArrowScalarFuncExpression(
                    src.empty_data,
                    [src],
                    "utf8_slice_codeunits",
                    (start, None, 1),
                )
            else:
                without_start_expr = src

            # Remove earlier occurrences so that extract_regex can find the correct occurrence/substring matching the regexp
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

        if func_name == "PI" and len(op_exprs) == 0:
            dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            return ConstantExpression(dummy_empty_data, input_plan, np.pi)

        if (
            func_name
            in (
                "ACOS",
                "ACOSH",
                "ASIN",
                "ASINH",
                "COS",
                "COSH",
                "SIN",
                "SINH",
                "TAN",
                "TANH",
                "ATAN",
                "ATANH",
            )
            and len(op_exprs) == 1
        ):
            src = op_exprs[0]
            # Arrow's Trigonometric functions return float32 for float32 input and
            # float64 for float64 and decimal input:
            # https://arrow.apache.org/docs/cpp/compute.html#trigonometric-functions
            src_dtype = src.empty_data.dtypes.iloc[0]
            out_dtype = pd.ArrowDtype(
                pa.float32()
                if src_dtype.pyarrow_dtype == pa.float32()
                else pa.float64()
            )
            dummy_empty_data = pd.Series(dtype=out_dtype)
            return ArrowScalarFuncExpression(
                dummy_empty_data,
                [src],
                func_name.lower(),
                (),
            )

        if func_name == "ATAN2" and len(op_exprs) == 2:
            src1 = op_exprs[0]
            src2 = op_exprs[1]
            src_dtype = src1.empty_data.dtypes.iloc[0]
            src2_dtype = src2.empty_data.dtypes.iloc[0]
            out_dtype = pd.ArrowDtype(
                pa.float32()
                if (
                    src_dtype.pyarrow_dtype == pa.float32()
                    and src2_dtype.pyarrow_dtype == pa.float32()
                )
                else pa.float64()
            )
            dummy_empty_data = pd.Series(dtype=out_dtype)
            return ArrowScalarFuncExpression(
                dummy_empty_data,
                [src1, src2],
                "atan2",
                (),
            )

        if func_name in ("RADIANS", "DEGREES") and len(op_exprs) == 1:
            src = op_exprs[0]
            # Return float32 for float32 input and float64 for float64 and decimal input
            src_dtype = src.empty_data.dtypes.iloc[0]
            out_dtype = pd.ArrowDtype(
                pa.float32()
                if src_dtype.pyarrow_dtype == pa.float32()
                else pa.float64()
            )
            dummy_empty_data = pd.Series(dtype=out_dtype)
            ceof_expr = ConstantExpression(
                dummy_empty_data,
                input_plan,
                (np.pi / 180.0) if func_name == "RADIANS" else (180.0 / np.pi),
            )
            return ArithOpExpression(
                dummy_empty_data,
                src,
                ceof_expr,
                "__mul__",
            )

        if func_name == "COT" and len(op_exprs) == 1:
            src = op_exprs[0]
            # Return float32 for float32 input and float64 for float64 and decimal input
            src_dtype = src.empty_data.dtypes.iloc[0]
            out_dtype = pd.ArrowDtype(
                pa.float32()
                if src_dtype.pyarrow_dtype == pa.float32()
                else pa.float64()
            )
            dummy_empty_data = pd.Series(dtype=out_dtype)
            # COT is defined as 1 / tan(x):
            # https://github.com/bodo-ai/Bodo/blob/d8a047024e8cfd12993c8ad4e8d781c4f2723348/BodoSQL/bodosql/kernels/trig_array_kernels.py#L251
            one_expr = ConstantExpression(
                dummy_empty_data,
                input_plan,
                1.0,
            )
            tan_expr = ArrowScalarFuncExpression(
                dummy_empty_data,
                [src],
                "tan",
                (),
            )
            return ArithOpExpression(
                dummy_empty_data,
                one_expr,
                tan_expr,
                "__truediv__",
            )

        # If we didn't match a supported basic function, fall through to NotImplemented
        raise NotImplementedError(
            f"SqlBasicFunction {func_name} not supported yet: " + java_call.toString()
        )

    if operator_class_name == "SqlNullPolicyFunction":
        operands = java_call.getOperands()
        op_exprs = [java_expr_to_python_expr(ctx, o, input_plan) for o in operands]
        func_name = op.getName().upper()

        if func_name in ("FLOOR", "CEIL") and len(op_exprs) in (1, 2):
            inp = op_exprs[0]
            ensure_type_of_expr(inp, func_name + " input", (int, float))

            inp_dtype = get_expr_dtype(inp, func_name + " input")

            if len(op_exprs) == 1:
                if compare_types(inp_dtype, int):
                    # If input is an integer, FLOOR/CEIL is a no-op
                    return inp
                else:
                    # If input is a float, return FLOOR(inp) or CEIL(inp) as normal
                    return UnaryOpExpression(inp.empty_data, inp, func_name.lower())
            else:
                # Snowflake / BodoSQL has a scale option that dictates how many digits
                # after the decimal point the input should be raised or lowered to.
                # DuckDB and Arrow have no such option, so we have to emulate it.
                scale_expr = op_exprs[1]
                ensure_type_of_expr(scale_expr, "scale_expr", int)

                int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
                float_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))

                ten_expr = ConstantExpression(int_empty_data, input_plan, 10)

                if compare_types(inp_dtype, int):
                    # If input is an int, we don't need to do anything for scale >= 0.
                    # If scale < 0, we can use a formula that allows us to retain
                    # the integer type throughout the calculation.
                    # Division is not ideal for ints because __floordiv__ truncates
                    # towards zero, and __truediv__ could result in loss of precision.

                    # For FLOOR, result = inp - (inp % 10^abs(scale)), minus 10^abs(scale)
                    #   if inp % 10^abs(scale) is negative.
                    # For CEIL, result = inp - (inp % 10^abs(scale)), plus 10^abs(scale)
                    #   if inp % 10^abs(scale) is positive.

                    zero_expr = ConstantExpression(int_empty_data, input_plan, 0)
                    bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
                    scale_negative = ComparisonOpExpression(
                        bool_empty_data, scale_expr, zero_expr, operator.lt
                    )

                    # Calculate inp % 10^abs(scale)
                    scale_magnitude = UnaryOpExpression(
                        scale_expr.empty_data, scale_expr, "abs"
                    )
                    power_of_10 = ArithOpExpression(
                        int_empty_data, ten_expr, scale_magnitude, "__pow__"
                    )
                    inp_remainder = ArithOpExpression(
                        inp.empty_data, inp, power_of_10, "__mod__"
                    )

                    # inp - inp % 10^abs(scale)
                    inp_minus_remainder = ArithOpExpression(
                        inp.empty_data, inp, inp_remainder, "__sub__"
                    )

                    if func_name == "FLOOR":
                        # FLOOR: Subtract power of 10 if remainder is negative (not including 0)
                        should_add_remainder = ComparisonOpExpression(
                            bool_empty_data, inp_remainder, zero_expr, operator.lt
                        )
                        adjusted_inp_minus_remainder = ArithOpExpression(
                            inp.empty_data, inp_minus_remainder, power_of_10, "__sub__"
                        )
                    else:
                        # CEIL: Add power of 10 if remainder is positive (not including 0)
                        should_add_remainder = ComparisonOpExpression(
                            bool_empty_data, inp_remainder, zero_expr, operator.gt
                        )
                        adjusted_inp_minus_remainder = ArithOpExpression(
                            inp.empty_data, inp_minus_remainder, power_of_10, "__add__"
                        )

                    # Final result for the scale < 0 case
                    negative_scale_result = CaseExpression(
                        inp.empty_data,
                        should_add_remainder,
                        adjusted_inp_minus_remainder,
                        inp_minus_remainder,
                    )

                    # For 0 or positive scale, we don't have to do anything.
                    return CaseExpression(
                        inp.empty_data, scale_negative, negative_scale_result, inp
                    )
                else:
                    # If input is a float, we do:
                    # result = [FLOOR/CEIL](inp * 10^scale) / 10^scale
                    power_of_10 = ArithOpExpression(
                        float_empty_data, ten_expr, scale_expr, "__pow__"
                    )
                    scaled_inp = ArithOpExpression(
                        float_empty_data, inp, power_of_10, "__mul__"
                    )
                    scaled_inp_rounded = UnaryOpExpression(
                        float_empty_data, scaled_inp, func_name.lower()
                    )
                    return ArithOpExpression(
                        inp.empty_data, scaled_inp_rounded, power_of_10, "__truediv__"
                    )
        elif func_name == "POW" and len(op_exprs) == 2:
            left = op_exprs[0]
            right = op_exprs[1]
            out_empty = left.empty_data.iloc[:, 0] ** right.empty_data.iloc[:, 0]
            return ArithOpExpression(out_empty, left, right, "__pow__")
        elif func_name == "SQUARE" and len(op_exprs) == 1:
            inp = op_exprs[0]
            ensure_type_of_expr(inp, "SQUARE input", (int, float))
            out_empty = inp.empty_data.iloc[:, 0] * inp.empty_data.iloc[:, 0]
            return ArithOpExpression(out_empty, inp, inp, "__mul__")
        elif func_name == "LOG2" and len(op_exprs) == 1:
            inp = op_exprs[0]
            ensure_type_of_expr(inp, "LOG2 input", (int, float))

            # Retain current float output type if input is a float,
            # otherwise use float64.
            inp_dtype = get_expr_dtype(inp, "LOG2 input")
            if compare_types(inp_dtype, int):
                out_empty = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            else:
                out_empty = inp.empty_data

            return UnaryOpExpression(out_empty, inp, "log2")
        elif func_name == "LOG" and len(op_exprs) == 1:
            # The SQL function LOG(x) is mapped to Calcite LOG(x),
            # which in Calcite and most systems means LN(x), but Bodo
            # defines SQL LOG(x) as LOG10(x). For now, we raise an
            # informative error while we decide on the best solution.
            raise ValueError(
                "LOG with 1 argument is not currently supported in the C++ backend. Consider using the unambiguous LN or LOG10 instead."
            )
        elif func_name == "LOG" and len(op_exprs) == 2:
            # We read the first argument as the operand and
            # the second argument as the base, which is different
            # than e.g. Snowflake.
            inp = op_exprs[0]
            base_expr = op_exprs[1]
            ensure_type_of_expr(inp, "LOG input", (int, float))
            ensure_type_of_expr(base_expr, "LOG base", (int, float))

            float_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))

            # Retain current float output type if input is a float,
            # otherwise use float64.
            inp_dtype = get_expr_dtype(inp, "LOG input")
            if compare_types(inp_dtype, int):
                out_empty = float_empty_data
            else:
                out_empty = inp.empty_data

            if isinstance(base_expr, ConstantExpression):
                # If the base is a constant, we can use shortcuts.

                # Use dedicated log functions for special bases (e, 10, 2)
                if base_expr.value == 10:
                    return UnaryOpExpression(out_empty, inp, "log10")
                elif base_expr.value == 2:
                    return UnaryOpExpression(out_empty, inp, "log2")
                elif np.isclose(base_expr.value, np.e):
                    return UnaryOpExpression(out_empty, inp, "ln")

                # If not a special base, calculate scalar ln(base)
                log_of_base_expr = ConstantExpression(
                    float_empty_data, input_plan, np.log(base_expr.value)
                )
            else:
                # Calculate ln(base)
                log_of_base_expr = UnaryOpExpression(float_empty_data, base_expr, "ln")

            # Use change of base formula: log_base_(x) = log(x) / log(base)
            log_of_inp = UnaryOpExpression(out_empty, inp, "ln")
            return ArithOpExpression(
                out_empty, log_of_inp, log_of_base_expr, "__truediv__"
            )

        elif func_name in ("DIV0", "DIV0NULL") and len(op_exprs) == 2:
            dividend_expr = op_exprs[0]
            divisor_expr = op_exprs[1]
            ensure_type_of_expr(dividend_expr, "dividend_expr", (int, float))
            ensure_type_of_expr(divisor_expr, "divisor_expr", (int, float))

            float_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            quotient_expr = ArithOpExpression(
                float_empty_data, dividend_expr, divisor_expr, "__truediv__"
            )

            # Return 0 if divisor is 0 (or NULL, for DIV0NULL);
            # otherwise, return the standard quotient.

            zero_expr = ConstantExpression(float_empty_data, input_plan, 0.0)
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            divisor_is_zero = ComparisonOpExpression(
                bool_empty_data, divisor_expr, zero_expr, operator.eq
            )

            if func_name == "DIV0NULL":
                divisor_is_null = UnaryOpExpression(
                    bool_empty_data, divisor_expr, "isnull"
                )
                invalid_divisor = ConjunctionOpExpression(
                    bool_empty_data, divisor_is_zero, divisor_is_null, "__or__"
                )
            else:
                invalid_divisor = divisor_is_zero

            div0_quotient = CaseExpression(
                float_empty_data, invalid_divisor, zero_expr, quotient_expr
            )

            # Ensure NULL is always returned if the dividend is NULL
            dividend_is_null = UnaryOpExpression(
                bool_empty_data, dividend_expr, "isnull"
            )
            return CaseExpression(
                float_empty_data,
                dividend_is_null,
                NullExpression(float_empty_data, input_plan, 0),
                div0_quotient,
            )
        elif func_name == "WIDTH_BUCKET" and len(op_exprs) == 4:
            """Get the bucket the input number would be in if we had a histogram with a certain contiguous value range and number of buckets"""
            numeric_expr = op_exprs[0]
            min_val_expr = op_exprs[1]  # inclusive
            max_val_expr = op_exprs[2]  # exclusive
            num_buckets_expr = op_exprs[3]

            ensure_type_of_expr(numeric_expr, "numeric_expr", (int, float))
            ensure_type_of_expr(min_val_expr, "min_val_expr", (int, float))
            ensure_type_of_expr(max_val_expr, "max_val_expr", (int, float))
            ensure_type_of_expr(num_buckets_expr, "num_buckets_expr", int)

            # The min value of the range must be strictly less than the max value
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            min_max_valid = ComparisonOpExpression(
                bool_empty_data, min_val_expr, max_val_expr, operator.lt
            )

            # The number of buckets must be 1 or more
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            zero_expr = ConstantExpression(int_empty_data, input_plan, 0)
            num_buckets_valid = ComparisonOpExpression(
                bool_empty_data, num_buckets_expr, zero_expr, operator.gt
            )

            valid_inputs = ConjunctionOpExpression(
                bool_empty_data, min_max_valid, num_buckets_valid, "__and__"
            )

            # Custom logic for get_common_int_type to ensure the result is signed after subtraction.
            # Technically this isn't necessary here since we won't be getting any negative
            # results, but it's a good example for future situations where this might be needed.
            def get_common_signed(
                expr1_width, expr2_width, expr1_is_signed, expr2_is_signed
            ):
                return True

            float_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            # Calculate the length of the range between min_val_expr and max_value_expr.
            # Note that we first determine the common integer type so that we can get the
            # appropriate empty_data for the operation.
            common_int_type = get_common_int_type(
                max_val_expr, min_val_expr, get_common_signed=get_common_signed
            )[0]
            range_length = ArithOpExpression(
                pd.Series(dtype=pd.ArrowDtype(common_int_type))
                if common_int_type is not None
                else float_empty_data,
                max_val_expr,
                min_val_expr,
                "__sub__",
            )

            # By doing numeric_expr - min_val_expr, we normalize the input to the range [0, max_val_expr - min_val_expr).
            # We can assume this since we check earlier for numeric_expr being less than min_val_expr or greater than max_val_expr.
            common_int_type = get_common_int_type(
                numeric_expr, min_val_expr, get_common_signed=get_common_signed
            )[0]
            normalized_input = ArithOpExpression(
                pd.Series(dtype=pd.ArrowDtype(common_int_type))
                if common_int_type is not None
                else float_empty_data,
                numeric_expr,
                min_val_expr,
                "__sub__",
            )

            # Get the position of numeric_expr in the range as a float in [0.0, 1.0).
            range_position = ArithOpExpression(
                float_empty_data, normalized_input, range_length, "__truediv__"
            )
            # Multiply by the number of buckets to scale to the range [0.0, num_buckets)
            scaled_range_position = ArithOpExpression(
                float_empty_data, range_position, num_buckets_expr, "__mul__"
            )

            # Get the 1-based integer bucket number.
            # We use floor(scaled_range_position) + 1 instead of ceil() so that 0.0 is mapped to a bucket number of 1.
            # Note that scaled_range_position cannot equal num_buckets_expr.
            bucket_number = UnaryOpExpression(
                int_empty_data, scaled_range_position, "floor"
            )
            one_expr = ConstantExpression(int_empty_data, input_plan, 1)
            bucket_number = ArithOpExpression(
                int_empty_data, bucket_number, one_expr, "__add__"
            )

            # Check if numeric_expr is outside of [min_val_expr, max_val_expr)
            val_below_range = ComparisonOpExpression(
                bool_empty_data, numeric_expr, min_val_expr, operator.lt
            )
            val_above_range = ComparisonOpExpression(
                bool_empty_data, numeric_expr, max_val_expr, operator.ge
            )
            # If numeric_expr < min_val_expr, return bucket number of 0.
            # If numeric_expr >= max_val_expr, return bucket number of num_buckets + 1.
            num_buckets_plus_one = ArithOpExpression(
                int_empty_data, num_buckets_expr, one_expr, "__add__"
            )
            final_bucket_number = CaseExpression(
                int_empty_data,
                val_below_range,
                zero_expr,
                CaseExpression(
                    int_empty_data, val_above_range, num_buckets_plus_one, bucket_number
                ),
            )

            return CaseExpression(
                int_empty_data,
                valid_inputs,
                final_bucket_number,
                NullExpression(int_empty_data, input_plan, 0),
            )

        elif (
            func_name in ("BITAND", "BITOR", "BITXOR", "BITSHIFTLEFT", "BITSHIFTRIGHT")
            and len(op_exprs) == 2
        ):
            left_expr = op_exprs[0]
            right_expr = op_exprs[1]

            ensure_type_of_expr(left_expr, "left_expr", int)
            ensure_type_of_expr(right_expr, "right_expr", int)

            empty_data = left_expr.empty_data
            cast_empty_data = None
            int64_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))

            if func_name == "BITAND":
                arrow_equivalent_func = "bit_wise_and"
            elif func_name == "BITOR":
                arrow_equivalent_func = "bit_wise_or"
            elif func_name == "BITXOR":
                arrow_equivalent_func = "bit_wise_xor"
            elif func_name == "BITSHIFTLEFT":
                arrow_equivalent_func = "shift_left"
            elif func_name == "BITSHIFTRIGHT":
                arrow_equivalent_func = "shift_right"

            if func_name in ("BITSHIFTLEFT", "BITSHIFTRIGHT"):
                left_opr_sql_type = operands[0].getType()
                SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName

                if left_opr_sql_type.getSqlTypeName().equals(SqlTypeName.BINARY):
                    # Cast right_expr to match the bit width and signedness of left_expr.
                    # This minimizes cast operations necessary, since if the types of left_expr and right_expr match, the final result will have the type of left_expr.
                    # This is what we want to retain the original bit width after shifting, for BINARY input.
                    # Effectively we discard bits that were shifted past the left end of the original precision input.
                    right_expr = CastExpression(left_expr.empty_data, right_expr)
                else:
                    # For INTEGER input:
                    # Ensure arguments are int64.
                    # We don't want shift_left to be limited by the precision of the
                    # input (at least for INTEGER input).
                    # If one argument is uint64, this can help avoid a mismatch that will cause an error coming from the Arrow compute function.
                    # We do the same for shift_right to be consistent with Snowflake (max bit width signed output).
                    if func_name == "BITSHIFTLEFT":
                        left_expr = CastExpression(int64_empty_data, left_expr)
                        # (Arrow can take care of casting right_expr in this case.)
                        # For bitshifting, result type should be INT64 to match Snowflake and output on C++ side.
                        empty_data = int64_empty_data
                    else:
                        # For shift_right, we have to be careful that left_expr keeps the original
                        # signedness so the proper right shift (logical or arithmetic) is performed.
                        if pa.types.is_signed_integer(
                            left_expr.empty_data.dtypes[
                                left_expr.empty_data.columns[0]
                            ].pyarrow_dtype
                        ):
                            shift_right_empty_data = int64_empty_data
                        else:
                            shift_right_empty_data = pd.Series(
                                dtype=pd.ArrowDtype(pa.uint64())
                            )
                        left_expr = CastExpression(shift_right_empty_data, left_expr)
                        # Make right_expr's type match left_expr so that Arrow's type unification doesn't
                        # make int64 the common type when left_expr is uint64, causing a cast failure
                        right_expr = CastExpression(shift_right_empty_data, right_expr)
                        empty_data = shift_right_empty_data
                        # Ensure result is signed if input was unsigned
                        cast_empty_data = int64_empty_data
            else:
                # For BITAND/BITOR/BIXOR:
                # Make sure the type of empty_data has the larger bit width of the two inputs.
                # Also unify the types of the inputs to that type.
                left_expr_dtype = left_expr.empty_data.dtypes[
                    left_expr.empty_data.columns[0]
                ].pyarrow_dtype
                right_expr_dtype = right_expr.empty_data.dtypes[
                    right_expr.empty_data.columns[0]
                ].pyarrow_dtype
                left_expr_signed = pa.types.is_signed_integer(left_expr_dtype)
                right_expr_signed = pa.types.is_signed_integer(right_expr_dtype)

                # Cast inputs to unsigned before doing the operation if at least one is unsigned.
                # Again, this is to work around a quirk of Arrow's type unification.
                empty_data_bit_width = max(
                    left_expr_dtype.bit_width, right_expr_dtype.bit_width
                )
                empty_data_signed = left_expr_signed and right_expr_signed
                target_dtype = pd.ArrowDtype(
                    eval(
                        f"pa.{'' if empty_data_signed else 'u'}int{empty_data_bit_width}()"
                    )
                )
                empty_data = pd.Series(dtype=target_dtype)
                left_expr = CastExpression(empty_data, left_expr)
                right_expr = CastExpression(empty_data, right_expr)
                # Return signed if at least one of the inputs are signed
                if left_expr_signed is not right_expr_signed:
                    cast_empty_data = pd.Series(
                        dtype=pd.ArrowDtype(eval(f"pa.int{empty_data_bit_width}()"))
                    )

            result = ArrowScalarFuncExpression(
                empty_data, [left_expr, right_expr], arrow_equivalent_func, ()
            )

            # Cast the result to the desired type if (for one reason or another)
            # the input types could not be aligned with the proper output type
            if cast_empty_data is not None:
                result = CastExpression(cast_empty_data, result)

            return result
        elif func_name == "BITNOT" and len(op_exprs) == 1:
            src = op_exprs[0]
            ensure_type_of_expr(src, "src", int)
            return ArrowScalarFuncExpression(src.empty_data, [src], "bit_wise_not", ())
        elif func_name == "GETBIT" and len(op_exprs) == 2:
            src = op_exprs[0]
            bit_num = op_exprs[1]

            ensure_type_of_expr(src, "src", int)
            ensure_type_of_expr(bit_num, "bit_num", int)

            # Do a bitwise AND on `src` and a bitmask that is only set on the requested bit position.
            # If the result is nonzero, the requested bit is 1, else it is 0.

            # We should operate on uint64.
            # We want Arrow's shift_left to set the most significant bit when
            # shifting 1 63 positions to the left, but this only happens when 1 is unsigned.

            uint64_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.uint64()))

            one_expr = ConstantExpression(uint64_empty_data, input_plan, 1)
            bit_num = CastExpression(uint64_empty_data, bit_num)
            bitmask = ArrowScalarFuncExpression(
                uint64_empty_data, [one_expr, bit_num], "shift_left", ()
            )

            # Cast `src` to uint64 to avoid problematic type unification in bit_wise_and.
            src = CastExpression(uint64_empty_data, src)
            src_with_mask = ArrowScalarFuncExpression(
                uint64_empty_data, [src, bitmask], "bit_wise_and", ()
            )

            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            zero_expr = ConstantExpression(uint64_empty_data, input_plan, 0)
            is_bit_set = ComparisonOpExpression(
                bool_empty_data, src_with_mask, zero_expr, operator.ne
            )

            # Use if_else instead of case_when so that nulls in is_bit_set propagate to the result
            return ArrowScalarFuncExpression(
                uint64_empty_data, [is_bit_set, one_expr, zero_expr], "if_else", ()
            )
        elif func_name == "LEFT" and len(op_exprs) == 2:
            # Implement LEFT as substr(0,...)
            src = op_exprs[0]
            len_expr = op_exprs[1]

            ensure_type_of_expr(src, "src", (str, pa.binary()))
            ensure_arg_is_const_expr_of_type(len_expr, "len_expr", int)

            out_empty = src.empty_data.iloc[:, 0]
            return ArrowScalarFuncExpression(
                out_empty, [src], "utf8_slice_codeunits", (0, len_expr.value, 1)
            )
        elif func_name == "RIGHT" and len(op_exprs) == 2:
            # Implement RIGHT as substr(-len,...)
            src = op_exprs[0]
            len_expr = op_exprs[1]

            ensure_type_of_expr(src, "src", (str, pa.binary()))
            ensure_arg_is_const_expr_of_type(len_expr, "len_expr", int)

            out_empty = src.empty_data.iloc[:, 0]
            return ArrowScalarFuncExpression(
                out_empty, [src], "utf8_slice_codeunits", (-len_expr.value, None, 1)
            )
        elif func_name == "STARTSWITH" and len(op_exprs) == 2:
            src = op_exprs[0]
            match_expr = op_exprs[1]

            ensure_type_of_expr(src, "src", (str, pa.binary()))
            ensure_arg_is_const_expr_of_type(
                match_expr, "match_expr", (str, pa.binary())
            )

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

            ensure_type_of_expr(src, "src", (str, pa.binary()))
            ensure_arg_is_const_expr_of_type(
                match_expr, "match_expr", (str, pa.binary())
            )

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

            ensure_type_of_expr(src, "src", (str, pa.binary()))
            ensure_arg_is_const_expr_of_type(
                match_expr, "match_expr", (str, pa.binary())
            )

            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return ArrowScalarFuncExpression(
                bool_empty_data,
                [src],
                "match_substring",
                (match_expr.value,),
            )
        elif func_name == "LENGTH" and len(op_exprs) == 1:
            src = op_exprs[0]
            ensure_type_of_expr(src, "src", (str, pa.binary()))
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            return ArrowScalarFuncExpression(
                int_empty_data,
                [src],
                "utf8_length",
                (),
            )
        elif func_name in ("INSTR", "CHARINDEX") and len(op_exprs) in (2, 3):
            if func_name == "INSTR":
                src = op_exprs[0]
                match_expr = op_exprs[1]
            else:
                # Substring to search for is the first parameter for POSITION/CHARINDEX
                src = op_exprs[1]
                match_expr = op_exprs[0]

            ensure_type_of_expr(src, "src", (str, pa.binary()))
            ensure_arg_is_const_expr_of_type(
                match_expr, "match_expr", (str, pa.binary())
            )
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            if len(match_expr.value) == 0:
                return ConstantExpression(int_empty_data, input_plan, 1)

            if len(op_exprs) == 3:
                start_expr = op_exprs[2]
                ensure_arg_is_const_expr_of_type(start_expr, "start_expr", int)

                if start_expr.value > 0:
                    start = start_expr.value - 1
                else:
                    start = start_expr.value
            else:
                start = 0

            if start > 0:
                # If start index is beyond the length of the string, we expect this to return an empty string
                without_start_expr = ArrowScalarFuncExpression(
                    src.empty_data,
                    [src],
                    "utf8_slice_codeunits",
                    (start, None, 1),
                )
            else:
                without_start_expr = src

            # Find the first occurrence of the substring in the sliced string
            substring_pos_expr = ArrowScalarFuncExpression(
                int_empty_data,
                [without_start_expr],
                "find_substring",
                (match_expr.value,),
            )

            # find_substring emits -1 when the substring is not found.
            # We need to return 0 in this case
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            negative_one_expr = ConstantExpression(int_empty_data, input_plan, -1)
            substring_not_found = ComparisonOpExpression(
                bool_empty_data, substring_pos_expr, negative_one_expr, operator.eq
            )

            # Add the ignored start index to the result only if substring was found
            zero_expr = ConstantExpression(int_empty_data, input_plan, 0)
            start_expr = ConstantExpression(int_empty_data, input_plan, start)
            offset_expr = CaseExpression(
                int_empty_data, substring_not_found, zero_expr, start_expr
            )
            adjusted_substring_pos_expr = ArithOpExpression(
                int_empty_data, substring_pos_expr, offset_expr, "__add__"
            )

            # Add 1 to find_substring expression since Arrow's find_substring is 0-indexed instead of 1-based like INSTR/CHARINDEX
            # If adjusted_substring_pos_expr is -1 then this will give the correct output of 0
            one_expr = ConstantExpression(int_empty_data, input_plan, 1)
            return ArithOpExpression(
                int_empty_data, adjusted_substring_pos_expr, one_expr, "__add__"
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
            ensure_type_of_expr(src, "src", (str, pa.binary()))

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
                    ensure_type_of_expr(
                        other_str_src, "other_str_src", (str, pa.binary())
                    )
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
            ensure_type_of_expr(separator, "separator", (str, pa.binary()))

            if len(op_exprs) == 2:
                # Nothing to concatenate, just return the input string
                return op_exprs[1]

            input_exprs = []
            for str_src in op_exprs[1:]:
                ensure_type_of_expr(str_src, "str_src", (str, pa.binary()))
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

            ensure_type_of_expr(src, "src", (str, pa.binary()))
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
            ensure_type_of_expr(src, "src", (str, pa.binary()))

            return ArrowScalarFuncExpression(src.empty_data, [src], "utf8_reverse", ())
        elif (
            func_name
            in (
                "ACOS",
                "ACOSH",
                "ASIN",
                "ASINH",
                "COS",
                "COSH",
                "SIN",
                "SINH",
                "TAN",
                "TANH",
                "ATAN",
                "ATANH",
            )
            and len(op_exprs) == 1
        ):
            src = op_exprs[0]
            # Arrow's Trigonometric functions return float32 for float32 input and
            # float64 for float64 and decimal input:
            # https://arrow.apache.org/docs/cpp/compute.html#trigonometric-functions
            src_dtype = src.empty_data.dtypes.iloc[0]
            out_dtype = pd.ArrowDtype(
                pa.float32()
                if src_dtype.pyarrow_dtype == pa.float32()
                else pa.float64()
            )
            dummy_empty_data = pd.Series(dtype=out_dtype)
            return ArrowScalarFuncExpression(
                dummy_empty_data,
                [src],
                func_name.lower(),
                (),
            )
        elif func_name == "RTRIMMED_LENGTH" and len(op_exprs) == 1:
            src = op_exprs[0]
            ensure_type_of_expr(src, "src", (str, pa.binary()))
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

            ensure_type_of_expr(src, "src", (str, pa.binary()))
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
        elif func_name == "CONVERT_TIMEZONE" and len(op_exprs) == 2:
            str_timezone = op_exprs[0]
            src = op_exprs[1]

            ensure_arg_is_const_expr_of_type(
                str_timezone, "str_timezone", (str, pa.binary())
            )
            ensure_type_of_expr(
                src,
                "src",
                (
                    pd._libs.tslibs.timestamps.Timestamp,
                    pd.ArrowDtype(pa.timestamp("ns")),
                ),
            )
            timestamp_pa_type = src.empty_data.iloc[:, 0].dtype.pyarrow_dtype
            target_res = "ns"
            # We figure out the right resolution to use.
            if pa.types.is_date(timestamp_pa_type):
                target_res = "s"
            elif pa.types.is_timestamp(timestamp_pa_type):
                target_res = timestamp_pa_type.unit

            # For now, if the resolution isn't our default, baked-in nanosecond
            # resolution then don't convert the operation because otherwise
            # tests will fail.
            if target_res == "ns":
                # The definition of this operation is to convert a time from one
                # timezone to a different one.
                # The general algorithm is that we get to a timestamp that has a timezone
                # and then use a cast to a different timezone which will actually perform
                # the conversion.  However, some input types don't have a timezone to
                # work with.  The check below find such cases and then uses the
                # assume_timezone kernel to apply the BodoSQL context's default_timezone
                # if it has one else use UTC.  One final wrinkle is that assume_timezone
                # can't operate on all possible input types so we convert the input
                # date/time type to a format that we know it can handle.
                if pa.types.is_date(timestamp_pa_type) or (
                    pa.types.is_timestamp(timestamp_pa_type)
                    and timestamp_pa_type.tz is None
                ):
                    tz = ctx.default_tz if ctx.default_tz is not None else "UTC"
                    local_timestamp_empty_data = pd.Series(
                        dtype=pd.ArrowDtype(pa.timestamp(target_res, tz=tz))
                    )
                    timestamp_empty_data_no_tz = pd.Series(
                        dtype=pd.ArrowDtype(pa.timestamp(target_res))
                    )
                    # We first cast to a data type that assume_timezone can work with.
                    src = CastExpression(timestamp_empty_data_no_tz, src)
                    # We use the context default_tz to make the timezone explicit.
                    src = ArrowScalarFuncExpression(
                        local_timestamp_empty_data,
                        [src],
                        "assume_timezone",
                        (tz,),
                    )

                target_timestamp_empty_data = pd.Series(
                    dtype=pd.ArrowDtype(pa.timestamp(target_res, tz=str_timezone.value))
                )
                # We use cast to convert timezones.
                target_timestamp_expr = CastExpression(target_timestamp_empty_data, src)
                return target_timestamp_expr

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
            sarg = search_expr.value
            nullAs = _get_sarg_null_as(sarg)

            def process_one_search_option(lower, lower_incl, upper, upper_incl):
                """Generate an expression to check if src satisfies this
                current possibility from the range set."""
                # The lower and upper bounds being equal is a special case that does not
                # involve less-than or greater-than comparison. It can be subdivided
                # based on whether the lower and upper bounds are inclusive or not.
                # The typical case is that they will be inclusive, which we can simplify
                # to an equality check.
                if lower is not None and upper is not None and lower == upper:
                    if lower_incl and upper_incl:
                        # Range of the form [a..a] - reduce to equality check
                        const_empty_data = arrow_to_empty_df(
                            pa.schema([pa.field("equal", pa.scalar(lower).type)])
                        )
                        return ComparisonOpExpression(
                            bool_empty_data,
                            src,
                            ConstantExpression(const_empty_data, input_plan, lower),
                            operator.eq,
                        )
                    elif lower_incl or upper_incl:
                        # Range of the form [a..a) or (a..a] - interpret as empty
                        return ConstantExpression(bool_empty_data, input_plan, False)
                    else:
                        raise ValueError("SEARCH option range form (a..a) is invalid.")

                # Address the standard continuous range case, e.g. BETWEEN
                in_range = None
                src_greater = None
                src_less = None
                if lower is not None:
                    const_empty_data = arrow_to_empty_df(
                        pa.schema([pa.field("equal", pa.scalar(lower).type)])
                    )
                    src_greater = ComparisonOpExpression(
                        bool_empty_data,
                        src,
                        ConstantExpression(const_empty_data, input_plan, lower),
                        operator.ge if lower_incl else operator.gt,
                    )
                    in_range = src_greater
                if upper is not None:
                    const_empty_data = arrow_to_empty_df(
                        pa.schema([pa.field("equal", pa.scalar(upper).type)])
                    )
                    src_less = ComparisonOpExpression(
                        bool_empty_data,
                        src,
                        ConstantExpression(const_empty_data, input_plan, upper),
                        operator.le if upper_incl else operator.lt,
                    )
                    in_range = src_less

                if lower is not None and upper is not None:
                    # Assure input is within both bounds
                    in_range = ConjunctionOpExpression(
                        bool_empty_data, src_greater, src_less, "__and__"
                    )
                elif lower is None and upper is None:
                    # No bounds specified, inputs must be in infinite range
                    in_range = ConstantExpression(bool_empty_data, input_plan, True)

                return in_range

            search_options = list(iter_sarg_ranges(sarg))
            out_expr = process_one_search_option(*search_options[0])
            # The definition of search is that the value is one of the
            # possibilities in the range set.  so, "or" in the other
            # possibilities below.
            for so in search_options[1:]:
                out_expr = ConjunctionOpExpression(
                    bool_empty_data, out_expr, process_one_search_option(*so), "__or__"
                )

            if nullAs != "UNKNOWN":
                # Replace nulls in the output with True or False depending on the value
                # of nullAs.
                out_expr = ArrowScalarFuncExpression(
                    bool_empty_data,
                    [
                        out_expr,
                        ConstantExpression(
                            bool_empty_data, input_plan, nullAs == "TRUE"
                        ),
                    ],
                    "coalesce",
                    (),
                )

            return out_expr

        raise NotImplementedError(
            f"Function name {func_name} not supported for SEARCH operator yet: "
            + java_call.toString()
        )

    if operator_class_name == "SqlRandomOperator":
        operands = java_call.getOperands()
        op_exprs = [java_expr_to_python_expr(ctx, o, input_plan) for o in operands]
        func_name = op.getName().upper()

        # NOTE: Calcite maps SQL RANDOM() to RANDOM() which means a random number
        # between [0.0, 1.0] in Calcite and does not accept a seed argument. For
        # completeness, to match Snowflake, we support a seed parameter anyway.
        if func_name == "RANDOM" and len(op_exprs) in (0, 1):
            """Generates random 64-bit integers"""
            int_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            # Create a dummy expression from the input plan which will be used on the
            # C++ side to check the row count (number of random values to generate).
            row_count_info_expr = ConstantExpression(int_empty_data, input_plan, 0)

            # Get and pass seed if given, else we rely on a system-generated seed
            # which is set on the C++ side.
            if len(op_exprs) == 1:
                seed_expr = op_exprs[0]
                # Should be a constant
                ensure_arg_is_const_expr_of_type(seed_expr, "seed_expr", int)
                seed_options = (seed_expr.value,)
            else:
                seed_options = ()

            # Arrow doesn't have any sort of randint function, so call our own that
            # has the name random_int64.
            return ArrowScalarFuncExpression(
                int_empty_data, [row_count_info_expr], "random_int64", seed_options
            )

    if operator_class_name == "SqlLeastGreatestFunction":
        operands = java_call.getOperands()
        op_exprs = [java_expr_to_python_expr(ctx, o, input_plan) for o in operands]
        func_name = op.getName().upper()
        assert func_name in ("LEAST", "GREATEST"), (
            "Unexpected function name for SqlLeastGreatestFunction: " + func_name
        )
        arrow_func = (
            "max_element_wise" if func_name == "GREATEST" else "min_element_wise"
        )
        # Check for supported data types in Arrow backend
        has_string = False
        has_nonstring = False
        for expr in op_exprs:
            expr_dtype = get_expr_dtype(expr)
            if compare_types(expr_dtype, str):
                has_string = True
            else:
                has_nonstring = True
            if compare_types(expr_dtype, bool):
                raise ValueError(f"Cannot use boolean types in {func_name} operator")
            if (
                isinstance(expr_dtype, pd.ArrowDtype)
                and pa.types.is_timestamp(expr_dtype.pyarrow_dtype)
                and expr_dtype.pyarrow_dtype.tz is not None
            ):
                raise ValueError(
                    f"Cannot use timezone-aware timestamp types in {func_name} operator"
                )
        # TODO(ehsan): cast strings to the datetime data type to match SQL semantics
        if has_string and has_nonstring:
            raise ValueError(
                f"Cannot mix string and non-string types in {func_name} operator"
            )
        # TODO(ehsan): get empty_data for the common type of the operands
        return ArrowScalarFuncExpression(
            op_exprs[0].empty_data, op_exprs, arrow_func, ()
        )

    raise NotImplementedError(
        f"Call operator {operator_class_name} not supported yet: "
        + java_call.toString()
    )


def compare_types(obj_type, expected_type):
    """
    Type checker that accounts for numpy/pandas/pyarrow dtypes.
    Returns True if `obj_type` is a subclass of `expected_type` or are otherwise considered equivalent
    """
    if isinstance(obj_type, str):
        try:
            obj_type = eval(obj_type)
        except Exception:
            pass
        obj_type = pd.api.types.pandas_dtype(obj_type)
    if isinstance(expected_type, str):
        try:
            expected_type = eval(expected_type)
        except Exception:
            pass
        expected_type = pd.api.types.pandas_dtype(expected_type)

    if isinstance(obj_type, type) and isinstance(expected_type, type):
        # NOTE: bool is a subclass of int in Python, but we don't want to consider them
        # equivalent for our purposes
        if issubclass(obj_type, expected_type) and not (
            obj_type is bool and expected_type is int
        ):
            return True

    if isinstance(obj_type, pa.DataType) and isinstance(expected_type, pa.DataType):
        if obj_type.equals(expected_type):
            return True
    # Convert pyarrow datatype objects to pandas.
    # This helps with compatibility with, e.g., pd.api.types.is_integer_dtype and the .numpy_dtype accessor below.
    if isinstance(obj_type, pa.DataType):
        # Note that wrapping pyarrow types in pd.ArrowDtype seems to behave better than pa.DataType.to_pandas_dtype()
        # For example, pa.string().to_pandas_dtype() just gives np.object_
        obj_type = pd.ArrowDtype(obj_type)
    if isinstance(expected_type, pa.DataType):
        expected_type = pd.ArrowDtype(expected_type)

    if expected_type is int:
        return pd.api.types.is_integer_dtype(obj_type)
    if expected_type is float:
        return pd.api.types.is_float_dtype(obj_type)
    if expected_type is str:
        return pd.api.types.is_string_dtype(obj_type)
    if expected_type is bool:
        return pd.api.types.is_bool_dtype(obj_type)

    if pd.api.types.is_dtype_equal(obj_type, expected_type):
        return True

    # Works for most pd.api.types.ExtensionDtypes
    # This can convert the nullable Pandas dtypes into standard numpy dtypes for easier comparison
    if hasattr(obj_type, "numpy_dtype"):
        obj_type = obj_type.numpy_dtype
    if hasattr(expected_type, "numpy_dtype"):
        expected_type = expected_type.numpy_dtype

    if np.issubdtype(obj_type, expected_type):
        return True

    return False


def ensure_arg_is_const_expr_of_type(expr, expr_name, dtype):
    if not isinstance(expr, bodo.pandas.plan.ConstantExpression):
        raise ValueError(
            f"{expr_name} should be ConstantExpression but instead was {type(expr)}"
        )

    if not isinstance(dtype, (list, tuple, set)):
        dtype = (dtype,)
    for dtype_alternative in dtype:
        if compare_types(type(expr.value), dtype_alternative):
            return
    if len(dtype) > 1:
        raise ValueError(
            f"{expr_name}.value should be one of {str(dtype)} but instead was {type(expr.value)}"
        )
    else:
        raise ValueError(
            f"{expr_name}.value should be {str(dtype[0])} but instead was {type(expr.value)}"
        )


def get_expr_dtype(expr, expr_name="Expression", get_const_val_type=True):
    """
    Get the type of the input `bodo.pandas.plan.Expression`.

    If `get_const_val_type` = `True` (the default), this will return the
    actual type of the constant value if `expr` is a `ConstantExpression`.
    Pass `get_const_val_type` = `False` to always get the empty_data type.
    """

    if not isinstance(expr, bodo.pandas.plan.Expression):
        return type(expr)

    if isinstance(expr, bodo.pandas.plan.ConstantExpression) and get_const_val_type:
        return type(expr.value)
    else:
        if isinstance(expr.empty_data, (pd.Series, np.ndarray)):
            return expr.empty_data.dtype
        elif isinstance(expr.empty_data, pd.DataFrame):
            assert len(expr.empty_data.columns) == 1
            return expr.empty_data.dtypes[expr.empty_data.columns[0]]
        else:
            raise ValueError(
                f"get_expr_dtype: Unsupported type of {expr_name}.empty_data:",
                type(expr.empty_data),
            )


def ensure_type_of_expr(expr, expr_name, dtype):
    expr_dtype = get_expr_dtype(expr, expr_name)

    if not isinstance(dtype, (list, tuple, set)):
        dtype = (dtype,)
    for dtype_alternative in dtype:
        if compare_types(expr_dtype, dtype_alternative):
            return

    if len(dtype) > 1:
        raise ValueError(
            f"Expected {expr_name} ({type(expr)}) to hold one of the datatypes {str(dtype)}, instead was {expr_dtype}"
        )
    else:
        raise ValueError(
            f"Expected {expr_name} ({type(expr)}) to hold datatype {str(dtype[0])}, instead was {expr_dtype}"
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
                ensure_type_of_expr(op_expr, "op_expr (|| arg)", (str, pa.binary()))

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


def get_common_int_type_list(
    exprs,
    get_common_width=lambda expr1_width,
    expr2_width,
    expr1_is_signed,
    expr2_is_signed: max(expr1_width, expr2_width),
    get_common_signed=lambda expr1_width,
    expr2_width,
    expr1_is_signed,
    expr2_is_signed: expr1_is_signed
    if expr1_is_signed == expr2_is_signed
    else (expr1_is_signed and expr1_width > expr2_width)
    or (expr2_is_signed and expr2_width > expr1_width),
):
    """Find a common integer type for a list of expressions with integer dtypes.

    `get_common_width` and `get_common_signed` are functions accepting the arguments
    `expr1_width`, `expr2_width`, `expr1_is_signed`, `expr2_is_signed`, and respectively
    returning what the common bit width and signedness of the expr1 and expr2
    integers should be.

    For the default `get_common_width()`: The bit width of the common type will be
    the maximum of the bit widths of the input types.

    For the default `get_common_signed()`: If all expressions have the same signedness,
    the signedness does not change. If expressions have different signedness, the
    common type will be unsigned unless the unsigned inputs have shorter bit widths
    than the signed inputs.

    Returns a tuple: `(common_arrow_type, cast_needed_list)` where each element in cast_needed_list
    is a boolean representing if the corresponding expr needs to be casted to match the common type.
    `(None, [False] * len(exprs))` is returned if any of the input exprs does not have integer dtype.
    """
    if not exprs:
        return None, []

    def get_as_pyarrow_dtype(dtype):
        if isinstance(dtype, pd.ArrowDtype):
            return dtype.pyarrow_dtype
        elif isinstance(dtype, pa.DataType):
            return dtype

        if not np.issubdtype(dtype, np.integer):
            dtype = pd.api.types.pandas_dtype(dtype)

            if hasattr(dtype, "numpy_dtype"):
                dtype = dtype.numpy_dtype
            else:
                if not np.issubdtype(dtype, np.integer):
                    raise ValueError(
                        f"get_as_py_arrow_dtype: unable to convert {dtype} to a numpy dtype"
                    )

        return pa.from_numpy_dtype(dtype)

    types = []
    for expr in exprs:
        dtype = get_expr_dtype(expr, get_const_val_type=False)
        if not compare_types(dtype, int):
            return None, [False] * len(exprs)
        types.append(get_as_pyarrow_dtype(dtype))

    common_type = types[0]

    for dtype in types[1:]:
        if common_type.equals(dtype):
            continue

        common_is_signed = pa.types.is_signed_integer(common_type)
        expr_is_signed = pa.types.is_signed_integer(dtype)
        common_width = common_type.bit_width
        expr_width = dtype.bit_width

        common_type_width = get_common_width(
            common_width, expr_width, common_is_signed, expr_is_signed
        )
        common_type_signed = get_common_signed(
            common_width, expr_width, common_is_signed, expr_is_signed
        )
        common_type = eval(
            f"pa.{'' if common_type_signed else 'u'}int{common_type_width}()"
        )

    cast_needed_list = [not expr_type.equals(common_type) for expr_type in types]

    return common_type, cast_needed_list


def get_common_int_type(left_expr, right_expr, *args, **kwargs):
    """Find a common integer type for two expressions with integer dtypes.

    See `get_common_int_type_list` for more details.

    Returns a tuple: `(common_arrow_type, left_cast_needed, right_cast_needed)`
    `(None, False, False)` is returned if `left_expr` or `right_expr` does not have integer dtype.
    """

    common_type, casts_needed = get_common_int_type_list(
        [left_expr, right_expr], *args, **kwargs
    )
    return (
        common_type,
        casts_needed[0],
        casts_needed[1],
    )


def make_unified_case_expression(empty_data, when_expr, then_expr, else_expr):
    """
    Make a DataFrame library CaseExpression with logic (`get_common_int_type`)
    to unify the integer types of `then_expr` and `else_expr` before passing
    to Arrow's case_when. Note that the output type (from `empty_data`) is
    retained. If `then_expr` or `else_expr` is not an integer expression,
    this is equivalent to directly constructing a CaseExpression.

    If `empty_data` is None or `"common"`, the output type will be the same as
    the common type of the inputs. If `then_expr` and `else_expr` have no
    common type, `empty_data` will default to `then_expr.empty_data`.
    """
    # then_expr and else_expr could have different types, e.g. int64 and uint64

    # Here we explicitly unify integer types to prevent Arrow's case_when
    # from attempting its own overflow-free unification which can fail in some cases,
    # notably for mixed int64 and uint64 inputs

    unified_then_expr = then_expr
    unified_else_expr = else_expr
    if isinstance(empty_data, (pd.Series, pd.DataFrame)):
        unified_empty_data = empty_data
    else:
        assert empty_data in (None, "common")
        unified_empty_data = then_expr.empty_data

    common_arrow_type, then_needs_cast, else_needs_cast = get_common_int_type(
        then_expr, else_expr
    )

    if common_arrow_type is not None:
        unified_empty_data = pd.Series(dtype=pd.ArrowDtype(common_arrow_type))

        # Wrap expressions in CastExpression if needed
        if then_needs_cast:
            unified_then_expr = CastExpression(unified_empty_data, then_expr)
        if else_needs_cast:
            unified_else_expr = CastExpression(unified_empty_data, else_expr)

        # Update empty_data to the unified type so schema matches the actual result
        # This prevents Arrow from trying to safely cast the result to a mismatched type

    case_expr = CaseExpression(
        unified_empty_data, when_expr, unified_then_expr, unified_else_expr
    )
    # Restore the original return type if the types of then_expr and else_expr were different
    # and our unification ended up casting one away from the intended result type.
    # We do this with a CastExpression instead of via CaseExpression empty_data
    # (which attempts to cast safely in _arrow_array_to_pd) so that integer overflow is allowed.
    if isinstance(empty_data, (pd.Series, pd.DataFrame)):
        if then_needs_cast or else_needs_cast:
            return CastExpression(empty_data, case_expr)
    else:
        assert empty_data in (None, "common")
    return case_expr


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

    # At the moment we choose then_expr to be the result type - is there a better way to decide?
    return make_unified_case_expression(
        then_expr.empty_data, when_expr, then_expr, else_expr
    )


def java_join_to_python_join(ctx, java_join):
    """Convert a BodoSQL Java join plan to a Python join plan."""

    join_info = java_join.analyzeCondition()
    join_info_cls = join_info.getClass()
    field = join_info_cls.getField("nonEquiConditions")
    nonEquiConds = field.get(join_info)

    left_keys, right_keys = join_info.keys()
    key_indices = list(zip(left_keys, right_keys))
    join_type = JavaJoinTypeToDuckDB(java_join.getJoinType())
    force_broadcast = java_join.getBroadcastBuildSide()

    ctx.join_filter_info[java_join.getJoinFilterID()] = (
        java_join.getOriginalJoinFilterKeyLocations(),
        right_keys,
    )

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


def java_rtjf_to_join_info(ctx, java_plan) -> JoinFilterInfo:
    """Convert a BodoSQL Java runtime join filter node to a Python runtime join filter
    info.
    """
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
    new_orig_build_keys = []
    for fid, eq_cols, is_first_cols in sorted_filter_data:
        if fid not in ctx.join_filter_info:
            raise ValueError(f"Join filter ID {fid} not found in join filter info")

        orig_key_locs, java_build_keys = ctx.join_filter_info[fid]
        if len(java_build_keys) != len(eq_cols):
            raise ValueError(
                f"Join filter ID {fid} has {len(java_build_keys)} original build keys but {len(eq_cols)} equality filter columns"
            )

        filter_cols = [-1] * len(eq_cols)
        is_first = [False] * len(is_first_cols)

        for loc_ind, key in enumerate(orig_key_locs):
            filter_cols[key] = eq_cols[loc_ind]
            is_first[key] = is_first_cols[loc_ind]

        # Each element in eq_cols corresponds to an item in JoinInfo.leftKeys,
        # which indicates an equality condition with the key from
        # JoinInfo.rightKeys at the same position. The order of the rightKeys/leftKeys
        # lists might change during column pruning, so we track the original key
        # column locations in BodoPhysicalJoin and reorder filter_cols to match the
        # order of the final join condition. After reordering, each key in filter_cols
        # will correspond to a filter derived from the key/column in JoinInfo.rightKeys
        # at the same position (the filter column can be -1 if that column is not
        # available yet).
        # The C++ JoinState creates a vector that's length is <num_build_keys> to store
        # the min/max values for each build key. The order of this vector should match
        # the order of the build keys in JoinInfo.rightKeys, which
        # is equivalent to the order that the build keys appear in the join condition.
        # Finally, we use the values at orig_build_key_cols to lookup the
        # min/max values from the JoinState for each filter that we
        # can push into I/O.
        # https://github.com/bodo-ai/Bodo/blob/f8cbfd4705e346a860fc4121c6735d9e8960d2c0/bodo/pandas/optimizer/runtime_join_filter.cpp#L282
        # https://github.com/bodo-ai/Bodo/blob/f8cbfd4705e346a860fc4121c6735d9e8960d2c0/bodo/pandas/_util.cpp#L1182
        # https://github.com/bodo-ai/Bodo/blob/0edd4715fdbb302f505962e3dcdf484f7e971c4a/bodo/libs/streaming/_join.cpp#L1360
        build_cols_idxs = list(range(len(java_build_keys)))

        new_filter_ids.append(fid)
        new_equality_filter_columns.append(filter_cols)
        new_equality_is_first_locations.append(is_first)
        new_orig_build_keys.append(build_cols_idxs)

    return JoinFilterInfo(
        filter_ids=new_filter_ids,
        equality_filter_columns=new_equality_filter_columns,
        orig_build_key_cols=new_orig_build_keys,
        equality_is_first_locations=new_equality_is_first_locations,
    )


def generate_runtime_join_filter(
    join_info: JoinFilterInfo, input_plan: LazyPlan
) -> LogicalJoinFilter:
    """Construct an instance of a runtime join filter plan node from the given join filter info and input plan."""
    return LogicalJoinFilter(
        input_plan.empty_data,
        input_plan,
        join_info.filter_ids,
        join_info.equality_filter_columns,
        join_info.equality_is_first_locations,
        join_info.orig_build_key_cols,
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

    if lit_type_name.equals(SqlTypeName.TIME):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.time64("ns")))
        # getValue2() returns an integer representing milliseconds since midnight
        val = pa.scalar(java_literal.getValue2() * 1000, pa.time64("us"))
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

    # SYMBOL is internal Calcite enum and needs supported specifically for each case.
    # Just avoiding errors here if input exprs are processed before the function itself.
    if lit_type_name.equals(SqlTypeName.SYMBOL):
        return None

    raise NotImplementedError(
        f"Literal type {lit_type_name.toString()} not supported yet"
    )


def get_java_symbol(java_symbol):
    """Extract the value of a Java SYMBOL or CHAR literal (e.g. date/time units)."""
    assert java_symbol.getClass().getSimpleName() == "RexLiteral", (
        "get_java_symbol: expected RexLiteral but got "
        + java_symbol.getClass().getSimpleName()
    )
    SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName

    if java_symbol.getTypeName().equals(SqlTypeName.CHAR):
        return java_symbol.getValue2()

    assert java_symbol.getTypeName().equals(SqlTypeName.SYMBOL), (
        "get_java_symbol: expected SYMBOL but got "
        + java_symbol.getTypeName().toString()
    )

    return java_symbol.getValue().toString()


def standardize_java_time_unit(fname, time_unit):
    """Convert time unit to a standardized form to simplify the code.
    For example, convert yy to year.
    """
    standardizeTimeUnit = gateway.jvm.com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.standardizeTimeUnit
    return standardizeTimeUnit(fname, time_unit, None)


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

    convert_na_to_value = {}
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
        elif func_name == "first":
            assert len(arg_cols) == 1, (
                f"Only single-argument {func_name} aggregations are supported"
            )
            in_type = input_plan.pa_schema.field(arg_cols[0]).type
            assert not isinstance(in_type, pa.lib.LargeListType)
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
        elif func_name == "count_if" and not func.hasFilter():
            func_name = "sum"
            out_type = pa.int64()
            # Calcite seems to prepare count_if by creating a column filled with
            # 1, 0, or NA.  An entry is 1 if the value in the original column
            # corresponds with true when cast to bool, 0 if the value corresponds
            # with false, and NA if the value is NA.  "sum" can then be used to
            # count the true/1 values.  However, if all values in the group are
            # NA then arrow gives a NA result but the SQL spec says the result
            # should be 0 so we have an extra fixup step where we register NA
            # entries in the output column for this aggregation to be set to 0.
            convert_na_to_value[len(out_types)] = 0
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

    def fill_na_column(df, col_idx, val):
        col = df.columns[col_idx]
        # Convert NA to the default value.
        df[col] = df[col].fillna(val)
        return df

    # Use the convert_na_to_value entries created when there
    # is a count_if aggregation to convert any NA entries to
    # 0.  We may add other aggregations that need to convert
    # NA to some default value other than 0 so made this data
    # structure hold the value to convert to so it could be
    # used when such cases arise.
    for index, na_value in convert_na_to_value.items():
        plan = gen_plan_via_bodo_dataframe(fill_na_column, plan, index, na_value)
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
        if name == "COUNT_IF":
            return "count_if"

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

    if kind.equals(SqlKind.ANY_VALUE) and len(argList) == 1:
        return "first"

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
        data.append([java_literal_to_python_const(ctx, e) for e in row])

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


def java_literal_to_python_const(ctx, java_literal):
    """Convert a BodoSQL Java literal to a Python constant value."""

    lit_expr = java_literal_to_python_literal(ctx, java_literal, None)
    assert isinstance(lit_expr, (ConstantExpression, NullExpression)), (
        "java_literal_to_python_const: Expected ConstantExpression or NullExpression"
    )

    if isinstance(lit_expr, NullExpression):
        return None

    return lit_expr.value


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
    if sql_type_name.equals(SqlTypeName.NULL):
        return pa.null()

    raise NotImplementedError(f"SQL type {sql_type_name.toString()} not supported yet")


def visit_iceberg_node(ctx, java_plan, read_info: IcebergReadInfo):
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
        visit_iceberg_node(ctx, input, read_info)
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
        visit_iceberg_node(ctx, input, read_info)
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
        visit_iceberg_node(ctx, input, read_info)
        return

    if java_class_name == "IcebergRuntimeJoinFilter":
        read_info.join_filter_info = java_rtjf_to_join_info(ctx, java_plan)
        input = java_plan.getInput()
        visit_iceberg_node(ctx, input, read_info)
        return

    raise NotImplementedError(
        f"Iceberg plan node {java_class_name} not supported yet in visit_iceberg_node"
    )


def generate_iceberg_read(read_info: IcebergReadInfo):
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

    plan, _, _ = build_iceberg_read_plan(
        # path_str has the schema in it so it's not needed in table id
        # TODO: update when supporting other catalog types
        full_table_path[-1],
        location=path_str,
        row_filter=row_filter,
        join_filter_info=read_info.join_filter_info,
        selected_fields=read_fields,
        limit=read_info.limit,
    )

    # Insert Runtime Join Filters on top of the read if needed
    if read_info.join_filter_info is not None:
        plan = generate_runtime_join_filter(read_info.join_filter_info, plan)

    return plan


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

        if kind.equals(SqlKind.IS_TRUE):
            return pie.EqualTo(input, True)

        if kind.equals(SqlKind.IS_FALSE):
            return pie.EqualTo(input, False)

    if operator_class_name == "SqlPrefixOperator" and len(java_call.getOperands()) == 1:
        kind = op.getKind()
        SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind
        if kind.equals(SqlKind.NOT):
            operand = java_expr_to_pyiceberg_expr(
                java_call.getOperands()[0], field_names
            )
            return pie.Not(operand)

    if operator_class_name == "SqlSearchOperator":
        return java_search_to_pyiceberg_expr(java_call, field_names)

    if operator_class_name == "SqlNullPolicyFunction":
        func_name = op.getName().upper()
        if func_name == "STARTSWITH" and len(java_call.getOperands()) == 2:
            operands = java_call.getOperands()
            ref = java_expr_to_pyiceberg_expr(operands[0], field_names)
            prefix = java_expr_to_pyiceberg_expr(operands[1], field_names)
            return pie.StartsWith(ref, prefix)

    raise NotImplementedError(
        f"Call operator {operator_class_name} for pyiceberg not supported yet: "
        + java_call.toString()
    )


def java_search_to_pyiceberg_expr(java_call, field_names):
    """Convert a Calcite SEARCH call (e.g. IN/BETWEEN expressed as
    SEARCH($col, Sarg[...])) to a PyIceberg expression.

    Calcite represents IN predicates as SEARCH against a Sarg of discrete
    points, and NOT IN as SEARCH against the complement. Range-based
    predicates (e.g. BETWEEN) become a Sarg with non-point ranges.
    """
    import pyiceberg.expressions as pie

    operands = java_call.getOperands()
    ref = java_expr_to_pyiceberg_expr(operands[0], field_names)
    sarg = operands[1].getValue()
    null_as = _get_sarg_null_as(sarg)

    # Collect each range as a Python value (point) or a comparison pair (range).
    points = []
    range_exprs = []
    for lower, lower_incl, upper, upper_incl in iter_sarg_ranges(sarg):
        if (
            lower is not None
            and upper is not None
            and lower == upper
            and lower_incl
            and upper_incl
        ):
            points.append(lower)
        elif (
            lower is not None
            and upper is not None
            and lower == upper
            and lower_incl != upper_incl
        ):
            # [a..a) or (a..a] is an empty range.
            range_exprs.append(pie.AlwaysFalse())
        else:
            range_exprs.append(
                _sarg_range_to_pyiceberg_expr(ref, lower, lower_incl, upper, upper_incl)
            )

    if sarg.isPoints() and points:
        expr = pie.In(ref, set(points))
    elif sarg.isComplementedPoints() and points:
        expr = pie.NotIn(ref, set(points))
    elif range_exprs:
        expr = range_exprs[0]
        for re_ in range_exprs[1:]:
            expr = pie.Or(expr, re_)
    elif points:
        # Mixed points and ranges (shouldn't happen for isPoints/isComplementedPoints,
        # but handle defensively): OR together In and range expressions.
        expr = pie.In(ref, set(points))
        for re_ in range_exprs:
            expr = pie.Or(expr, re_)
    else:
        raise NotImplementedError(
            "SEARCH with empty Sarg not supported yet: " + java_call.toString()
        )

    if null_as == "FALSE":
        return pie.And(expr, pie.NotNull(ref))
    if null_as == "TRUE":
        return pie.Or(expr, pie.IsNull(ref))
    return expr


def _sarg_endpoint_to_python(endpoint):
    """Convert a Java Sarg range endpoint (e.g. NlsString, BigDecimal) to a
    Python value."""
    if isinstance(endpoint, py4j.java_gateway.JavaObject):
        # NlsString and other Calcite literal wrappers expose getValue()
        return endpoint.getValue()
    if isinstance(endpoint, decimal.Decimal):
        return float(endpoint)
    return endpoint


def _get_sarg_null_as(sarg):
    """Read the nullAs field of a Java Sarg as a string ('UNKNOWN', 'TRUE', or 'FALSE')."""
    return sarg.getClass().getDeclaredField("nullAs").get(sarg).toString()


def iter_sarg_ranges(sarg):
    """Iterate over the ranges in a Calcite Sarg's range set, yielding
    ``(lower, lower_inclusive, upper, upper_inclusive)`` tuples with
    Python-typed endpoints.

    A ``None`` bound means unbounded on that side. Calcite represents IN
    predicates as a Sarg of discrete points (each a degenerate [a..a]
    range), NOT IN as the complement, and range-based predicates (e.g.
    BETWEEN) as non-degenerate ranges.
    """
    range_set = sarg.getClass().getDeclaredField("rangeSet").get(sarg)
    it = range_set.asRanges().iterator()
    while it.hasNext():
        r = it.next()
        has_lower = r.hasLowerBound()
        has_upper = r.hasUpperBound()
        yield (
            _sarg_endpoint_to_python(r.lowerEndpoint()) if has_lower else None,
            r.lowerBoundType().toString() == "CLOSED" if has_lower else False,
            _sarg_endpoint_to_python(r.upperEndpoint()) if has_upper else None,
            r.upperBoundType().toString() == "CLOSED" if has_upper else False,
        )


def _sarg_range_to_pyiceberg_expr(ref, lower, lower_inclusive, upper, upper_inclusive):
    """Convert a single non-point Sarg range to a PyIceberg comparison
    expression against the given reference."""
    import pyiceberg.expressions as pie

    has_lower = lower is not None
    has_upper = upper is not None

    if has_lower and has_upper:
        left = (
            pie.GreaterThanOrEqual(ref, lower)
            if lower_inclusive
            else pie.GreaterThan(ref, lower)
        )
        right = (
            pie.LessThanOrEqual(ref, upper)
            if upper_inclusive
            else pie.LessThan(ref, upper)
        )
        return pie.And(left, right)
    if has_lower:
        return (
            pie.GreaterThanOrEqual(ref, lower)
            if lower_inclusive
            else pie.GreaterThan(ref, lower)
        )
    if has_upper:
        return (
            pie.LessThanOrEqual(ref, upper)
            if upper_inclusive
            else pie.LessThan(ref, upper)
        )
    # No bounds => infinite range, always true.
    return pie.AlwaysTrue()


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

    if kind.equals(SqlKind.IS_DISTINCT_FROM):
        # Null-safe "not equal": A != B AND (A IS NOT NULL OR B IS NOT NULL).
        # pyiceberg's NotEqualTo follows SQL semantics (null if either is
        # null), so the additional IS_NOT_NULL clause distinguishes the
        # "one is null" case from the "both null" case.
        left_nn = pie.NotNull(left)
        right_nn = pie.NotNull(right)
        return pie.And(pie.NotEqualTo(left, right), pie.Or(left_nn, right_nn))

    if kind.equals(SqlKind.IS_NOT_DISTINCT_FROM):
        # Null-safe "equal": A == B OR (A IS NULL AND B IS NULL).
        # pyiceberg's EqualTo follows SQL semantics (null if either is null),
        # so the additional IS_NULL clause covers the "both null" case.
        left_null = pie.IsNull(left)
        right_null = pie.IsNull(right)
        return pie.Or(pie.EqualTo(left, right), pie.And(left_null, right_null))

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

    if lit_type_name.equals(SqlTypeName.DECIMAL):
        lit_type_scale = lit_type.getScale()
        val = java_literal.getValue()
        if lit_type_scale == 0:
            return int(val)
        else:
            return val

    if lit_type_name.equals(SqlTypeName.DOUBLE):
        return java_literal.getValue()

    if lit_type_name.equals(SqlTypeName.FLOAT):
        return float(java_literal.getValue())

    if (
        lit_type_name.equals(SqlTypeName.TINYINT)
        or lit_type_name.equals(SqlTypeName.SMALLINT)
        or lit_type_name.equals(SqlTypeName.INTEGER)
        or lit_type_name.equals(SqlTypeName.BIGINT)
    ):
        return int(java_literal.getValue2())

    if lit_type_name.equals(SqlTypeName.BOOLEAN):
        return bool(java_literal.getValue())

    if lit_type_name.equals(SqlTypeName.CHAR):
        return java_literal.getValue2()

    if lit_type_name.equals(SqlTypeName.VARCHAR):
        return java_literal.getValue2()

    if lit_type_name.equals(SqlTypeName.DATE):
        # getValue2() returns an integer representing days since epoch
        return date(1970, 1, 1) + timedelta(days=java_literal.getValue2())

    if lit_type_name.equals(SqlTypeName.TIMESTAMP):
        # getValue2() returns an integer representing milliseconds since epoch
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
            milliseconds=java_literal.getValue2()
        )

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
