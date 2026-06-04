from __future__ import annotations

import operator
import zoneinfo
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import pyarrow as pa

import bodo
import bodo.pandas as bd
import bodosql
from bodo.pandas.plan import (
    AggregateExpression,
    ArithOpExpression,
    ArrowScalarFuncExpression,
    CaseExpression,
    CastExpression,
    ComparisonOpExpression,
    ConjunctionOpExpression,
    ConstantExpression,
    LogicalAggregate,
    LogicalComparisonJoin,
    LogicalCrossProduct,
    LogicalDistinct,
    LogicalFilter,
    LogicalJoinFilter,
    LogicalOrder,
    LogicalProjection,
    NullExpression,
    UnaryOpExpression,
    arrow_to_empty_df,
    make_col_ref_exprs,
)
from bodosql.imported_java_classes import JavaEntryPoint, gateway


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

    if operator_class_name == "SqlNullPolicyFunction":
        func_name = op.getName().upper()
        num_operands = len(java_call.getOperands())

        # Date part functions wrapped in SqlNullPolicyFunction (e.g. WEEKDAY($0))
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
        }
        if func_name in _DATE_PART_ARROW_FUNCS and num_operands == 1:
            input = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            arrow_func = _DATE_PART_ARROW_FUNCS[func_name]
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            return ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())

        if func_name == "DAYNAME" and num_operands == 1:
            input = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.string()))
            return ArrowScalarFuncExpression(empty_data, [input], "strftime", ("%a",))

        if func_name == "MONTHNAME" and num_operands == 1:
            input = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.string()))
            return ArrowScalarFuncExpression(empty_data, [input], "strftime", ("%b",))

        if func_name == "MONTH_NAME" and num_operands == 1:
            input = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.string()))
            return ArrowScalarFuncExpression(empty_data, [input], "strftime", ("%b",))

        if func_name == "MAKEDATE" and num_operands == 2:
            # MAKEDATE(year, dayofyear) → Jan 1 of year + (doy-1) days
            year_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[0], input_plan
            )
            doy_expr = java_expr_to_python_expr(
                ctx, java_call.getOperands()[1], input_plan
            )
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
            _TRUNC_UNIT_MAP = {
                "YEAR": "year",
                "QUARTER": "quarter",
                "MONTH": "month",
                "WEEK": "week",
                "DAY": "day",
                "HOUR": "hour",
                "MINUTE": "minute",
                "SECOND": "second",
            }
            arrow_unit = _TRUNC_UNIT_MAP.get(unit_raw, unit_raw.lower())
            input = java_expr_to_python_expr(
                ctx, java_call.getOperands()[1], input_plan
            )
            return ArrowScalarFuncExpression(
                input.empty_data, [input], "floor_temporal", (1, arrow_unit)
            )

        if func_name in ("DATEADD", "DATE_ADD", "ADDDATE"):
            # DATE_ADD(date, interval) or DATE_ADD(unit, amount, date)
            # For 2 operands: (date, interval) → date + interval
            # For 3 operands: (unit, amount, date) → date + (unit * amount)
            if num_operands == 2:
                return java_binop_to_python_expr(
                    ctx,
                    SqlKind.PLUS,
                    [
                        java_expr_to_python_expr(
                            ctx, java_call.getOperands()[i], input_plan
                        )
                        for i in range(num_operands)
                    ],
                )
            elif num_operands == 3:
                # DATEADD(unit_flag, amount, date) → date + timedelta(days=amount)
                # In Snowflake, date + integer always means date + N days.
                amount_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[1], input_plan
                )
                date_expr = java_expr_to_python_expr(
                    ctx, java_call.getOperands()[2], input_plan
                )
                interval_val = pd.Timedelta(days=int(amount_expr.value))
                dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.duration("ns")))
                interval_expr = ConstantExpression(
                    dummy_empty_data, input_plan, interval_val
                )
                out_empty = (
                    date_expr.empty_data.iloc[:, 0]
                    + interval_expr.empty_data.iloc[:, 0]
                )
                return ArithOpExpression(out_empty, date_expr, interval_expr, "__add__")
        if func_name in ("DATE_SUB", "SUBDATE"):
            # DATE_SUB(date, interval) or DATE_SUB(unit, amount, date)
            if num_operands == 2:
                return java_binop_to_python_expr(
                    ctx,
                    SqlKind.MINUS,
                    [
                        java_expr_to_python_expr(
                            ctx, java_call.getOperands()[i], input_plan
                        )
                        for i in range(num_operands)
                    ],
                )
            elif num_operands == 3:
                return java_binop_to_python_expr(
                    ctx,
                    SqlKind.MINUS,
                    [
                        java_expr_to_python_expr(
                            ctx, java_call.getOperands()[2], input_plan
                        ),
                        java_expr_to_python_expr(
                            ctx, java_call.getOperands()[1], input_plan
                        ),
                    ],
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
        return java_binop_to_python_expr(ctx, kind, op_exprs)

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
        # Strip "FLAG(" / ")" or "INTERVAL_" prefix from unit string
        if "(" in unit_str:
            unit_str = unit_str.split("(")[1].rstrip(")")
        _DATE_PART_ARROW_FUNCS_EXTRA = {
            "YEAR": "year",
            "MONTH": "month",
            "DAY": "day",
            "HOUR": "hour",
            "MINUTE": "minute",
            "SECOND": "second",
            "QUARTER": "quarter",
            "WEEK": "iso_week",
            "DOW": "day_of_week",
            "DOY": "day_of_year",
        }
        arrow_func = _DATE_PART_ARROW_FUNCS_EXTRA.get(unit_str)
        if arrow_func is None:
            raise NotImplementedError(f"Unsupported EXTRACT unit: {unit_str}")
        empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
        return ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())
    if (
        operator_class_name == "SqlDatePartFunction"
        and len(java_call.getOperands()) == 1
    ):
        operands = java_call.getOperands()
        input = java_expr_to_python_expr(ctx, operands[0], input_plan)
        func_name = op.getName().upper()

        # Map Calcite function names to Arrow compute function names
        _DATE_PART_ARROW_FUNCS = {
            "YEAR": "year",
            "MONTH": "month",
            "DAY": "day",
            "DAYOFMONTH": "day",
            "HOUR": "hour",
            "MINUTE": "minute",
            "SECOND": "second",
            "QUARTER": "quarter",
            "MICROSECOND": "microsecond",
            "NANOSECOND": "nanosecond",
            "WEEK": "iso_week",
            "WEEKOFYEAR": "iso_week",
            "WEEKISO": "iso_week",
            "DAYOFYEAR": "day_of_year",
            "DAYOFWEEK": "day_of_week",
            "WEEKDAY": "day_of_week",
        }
        arrow_func = _DATE_PART_ARROW_FUNCS.get(func_name, func_name.lower())

        if func_name in (
            "YEAR",
            "MONTH",
            "DAY",
            "DAYOFMONTH",
            "HOUR",
            "MINUTE",
            "SECOND",
            "QUARTER",
            "MICROSECOND",
            "NANOSECOND",
        ):
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            return ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())

        if func_name in ("WEEK", "WEEKOFYEAR", "WEEKISO"):
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            return ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())

        if func_name == "DAYOFYEAR":
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            return ArrowScalarFuncExpression(empty_data, [input], arrow_func, ())

        if func_name in ("DAYOFWEEK", "WEEKDAY"):
            empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
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
            dummy_empty_data = pd.Series(
                [curr_ts], dtype=pd.ArrowDtype(pa.timestamp("ns"))
            )
            return ConstantExpression(dummy_empty_data, input_plan, curr_ts)

    if operator_class_name == "SqlBasicFunction":
        # Map Calcite basic functions to Bodo expressions
        operands = java_call.getOperands()
        op_exprs = [java_expr_to_python_expr(ctx, o, input_plan) for o in operands]
        # function name as string (e.g., "POWER", "SQRT")
        func_name = op.getName().upper()

        if func_name in ("UTC_TIMESTAMP", "UTC_DATE"):
            curr_ts = pd.Timestamp.now(tz="UTC")
            if func_name == "UTC_DATE":
                curr_ts = curr_ts.normalize()
            dummy_empty_data = pd.Series(
                [curr_ts], dtype=pd.ArrowDtype(pa.timestamp("ns", tz="UTC"))
            )
            return ConstantExpression(dummy_empty_data, input_plan, curr_ts)

        if func_name in (
            "CURRENT_TIMESTAMP",
            "GETDATE",
            "LOCALTIMESTAMP",
            "SYSTIMESTAMP",
            "NOW",
        ):
            curr_ts = pd.Timestamp.now()
            dummy_empty_data = pd.Series(
                [curr_ts], dtype=pd.ArrowDtype(pa.timestamp("ns"))
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
                val = expr.value
                if isinstance(val, tuple) and len(val) == 2:
                    total_months += val[0]
                    total_nanos += val[1]
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
                combined_val = (total_months, total_nanos)
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

        if func_name == "IFF" and len(op_exprs) == 3:
            # IFF is equivalent to CASE with single WHEN
            return java_case_to_python_case(ctx, operands, input_plan)

        if func_name == "NULLIF" and len(op_exprs) == 2:
            return ArrowScalarFuncExpression(
                op_exprs[0].empty_data, op_exprs, "nullif", ()
            )

        # If we didn't match a supported basic function, fall through to NotImplemented
        raise NotImplementedError(
            f"SqlBasicFunction {func_name} not supported yet: " + java_call.toString()
        )

    raise NotImplementedError(
        f"Call operator {operator_class_name} not supported yet: "
        + java_call.toString()
    )


def java_binop_to_python_expr(ctx, kind, op_exprs):
    """Convert a BodoSQL Java binary operator call to a DataFrame library expression."""

    left = op_exprs[0]

    # Calcite may add more than 2 operand for the same binary operator
    if len(op_exprs) > 2:
        right = java_binop_to_python_expr(ctx, kind, op_exprs[1:])
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

    left_plan = java_plan_to_python_plan(ctx, java_join.getLeft())
    right_plan = java_plan_to_python_plan(ctx, java_join.getRight())

    empty_join_out = pd.concat([left_plan.empty_data, right_plan.empty_data], axis=1)
    empty_join_out.columns = java_join.getRowType().getFieldNames()

    if len(key_indices) > 0:
        # TODO[BSE-5150]: support broadcast join flag
        planJoinOrCross = LogicalComparisonJoin(
            empty_join_out,
            left_plan,
            right_plan,
            join_type,
            key_indices,
            java_join.getJoinFilterID(),
        )
    else:
        planJoinOrCross = LogicalCrossProduct(empty_join_out, left_plan, right_plan)

    if len(nonEquiConds) == 0:
        return planJoinOrCross
    else:
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

    for func in aggCallList:
        if func.hasFilter():
            raise NotImplementedError("Filtered aggregations are not supported yet")
        func_name = _agg_to_func_name(func)
        arg_cols = list(func.getArgList())
        if func_name == "size":
            assert len(arg_cols) in [0, 1], (
                "Size aggregations arg len not in [0,1] are not supported"
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

    names = list(java_plan.getRowType().getFieldNames())
    new_schema = pa.schema([pa.field(name, t) for name, t in zip(names, out_types)])
    empty_out_data = arrow_to_empty_df(new_schema)

    plan = LogicalAggregate(
        empty_out_data,
        input_plan,
        keys,
        exprs,
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

    if java_plan.getFetch() is not None or java_plan.getOffset() is not None:
        raise NotImplementedError("LIMIT/OFFSET in sort not supported yet")

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

    sorted_plan = LogicalOrder(
        input_plan.empty_data,
        input_plan,
        ascending,
        na_position,
        key_col_inds,
        input_plan.pa_schema,
    )
    return sorted_plan


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
