from __future__ import annotations

import decimal
import operator

import pandas as pd
import pyarrow as pa

import bodo
import bodo.pandas as bd
import bodosql
from bodo.pandas.plan import (
    ArithOpExpression,
    ComparisonOpExpression,
    ConstantExpression,
    LogicalComparisonJoin,
    LogicalFilter,
    LogicalProjection,
    arrow_to_empty_df,
    make_col_ref_exprs,
)
from bodosql.imported_java_classes import JavaEntryPoint, gateway


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
        elif isinstance(table, pd.DataFrame):
            return bodo.pandas.from_pandas(table)._plan
        else:
            raise NotImplementedError(
                f"Table type {type(table)} not supported in C++ backend yet"
            )

    if java_class_name in ("PandasProject", "BodoPhysicalProject"):
        input_plan = java_plan_to_python_plan(ctx, java_plan.getInput())
        exprs = [
            java_expr_to_python_expr(e, input_plan) for e in java_plan.getProjects()
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

    # TODO[BSE-5152]: support runtime join filters (ok to ignore for now since they are
    # just optimizations)
    if java_class_name == "BodoPhysicalRuntimeJoinFilter":
        return java_plan_to_python_plan(ctx, java_plan.getInput())

    if java_class_name == "BodoPhysicalFilter":
        return java_filter_to_python_filter(ctx, java_plan)

    raise NotImplementedError(f"Plan node {java_class_name} not supported yet")


def java_expr_to_python_expr(java_expr, input_plan):
    """Convert a BodoSQL Java expression to a DataFrame library expression
    (bodo.pandas.plan.Expression).
    """
    java_class_name = java_expr.getClass().getSimpleName()

    if java_class_name == "RexInputRef":
        col_index = java_expr.getIndex()
        return make_col_ref_exprs([col_index], input_plan)[0]

    if java_class_name == "RexCall":
        return java_call_to_python_call(java_expr, input_plan)

    if java_class_name == "RexLiteral":
        return java_literal_to_python_literal(java_expr, input_plan)

    raise NotImplementedError(f"Expression {java_class_name} not supported yet")


def java_call_to_python_call(java_call, input_plan):
    """Convert a BodoSQL Java call expression to a DataFrame library expression
    (bodo.pandas.plan.Expression).
    """
    op = java_call.getOperator()
    operator_class_name = op.getClass().getSimpleName()

    if (
        operator_class_name in ("SqlMonotonicBinaryOperator", "SqlBinaryOperator")
        and len(java_call.getOperands()) == 2
    ):
        operands = java_call.getOperands()
        left = java_expr_to_python_expr(operands[0], input_plan)
        right = java_expr_to_python_expr(operands[1], input_plan)
        kind = op.getKind()
        SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind

        if kind.equals(SqlKind.PLUS):
            # TODO: support all BodoSQL data types in backend (including date/time)
            # TODO: upcast output to avoid overflow?
            expr = ArithOpExpression(left.empty_data, left, right, "__add__")
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

    if operator_class_name == "SqlCastFunction" and len(java_call.getOperands()) == 1:
        operand = java_call.getOperands()[0]
        operand_type = operand.getType()
        target_type = java_call.getType()
        SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName
        # TODO: support all Calcite casts

        if target_type.getSqlTypeName().equals(SqlTypeName.DECIMAL) and is_int_type(
            operand_type
        ):
            # Cast of int to DECIMAL is unnecessary in C++ backend
            return java_expr_to_python_expr(operand, input_plan)

    raise NotImplementedError(f"Call operator {operator_class_name} not supported yet")


def java_join_to_python_join(ctx, java_join):
    """Convert a BodoSQL Java join plan to a Python join plan."""
    from bodo.ext import plan_optimizer

    join_info = java_join.analyzeCondition()

    # TODO[BSE-5149]: support non-equi joins
    if not join_info.isEqui():
        raise NotImplementedError("Only equi-joins are supported")

    left_keys, right_keys = join_info.keys()
    key_indices = list(zip(left_keys, right_keys))
    is_left = java_join.getJoinType().generatesNullsOnLeft()
    is_right = java_join.getJoinType().generatesNullsOnRight()
    join_type = plan_optimizer.CJoinType.INNER
    if is_left and is_right:
        join_type = plan_optimizer.CJoinType.OUTER
    elif is_left:
        join_type = plan_optimizer.CJoinType.LEFT
    elif is_right:
        join_type = plan_optimizer.CJoinType.RIGHT

    left_plan = java_plan_to_python_plan(ctx, java_join.getLeft())
    right_plan = java_plan_to_python_plan(ctx, java_join.getRight())

    empty_join_out = pd.concat([left_plan.empty_data, right_plan.empty_data], axis=1)
    # Avoid duplicate column names
    empty_join_out.columns = [c + str(i) for i, c in enumerate(empty_join_out.columns)]

    # TODO[BSE-5150]: support broadcast join flag
    planComparisonJoin = LogicalComparisonJoin(
        empty_join_out,
        left_plan,
        right_plan,
        join_type,
        key_indices,
    )
    return planComparisonJoin


def java_filter_to_python_filter(ctx, java_filter):
    """Convert a BodoSQL Java filter plan to a Python filter plan."""
    input_plan = java_plan_to_python_plan(ctx, java_filter.getInput())
    condition = java_expr_to_python_expr(java_filter.getCondition(), input_plan)
    return LogicalFilter(input_plan.empty_data, input_plan, condition)


def java_literal_to_python_literal(java_literal, input_plan):
    """Convert a BodoSQL Java literal expression to a DataFrame library constant"""
    SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName
    lit_type_name = java_literal.getTypeName()

    # TODO: support all Calcite literal types

    if lit_type_name.equals(SqlTypeName.DECIMAL):
        # Integer constants are represented as DECIMAL in Calcite
        val = java_literal.getValue()
        if isinstance(val, decimal.Decimal) and val == int(val):
            dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            return ConstantExpression(dummy_empty_data, input_plan, int(val))

    if lit_type_name.equals(SqlTypeName.DOUBLE):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
        return ConstantExpression(dummy_empty_data, input_plan, java_literal.getValue())

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
