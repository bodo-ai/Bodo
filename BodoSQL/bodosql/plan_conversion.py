from __future__ import annotations

import pandas as pd
import pyarrow as pa

import bodo
import bodo.pandas as bd
import bodosql
from bodo.pandas.plan import (
    ArithOpExpression,
    LogicalComparisonJoin,
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
        new_schema = pa.schema([e.pa_schema.field(0) for e in exprs])
        empty_data = arrow_to_empty_df(new_schema)
        proj_plan = LogicalProjection(
            empty_data,
            input_plan,
            exprs,
        )
        return proj_plan

    if java_class_name == "BodoPhysicalJoin":
        return java_join_to_python_join(ctx, java_plan)

    # TODO: support runtime join filters (ok to ignore for now since they are just
    # optimizations)
    if java_class_name == "BodoPhysicalRuntimeJoinFilter":
        return java_plan_to_python_plan(ctx, java_plan.getInput())

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

    raise NotImplementedError(f"Expression {java_class_name} not supported yet")


def java_call_to_python_call(java_call, input_plan):
    """Convert a BodoSQL Java call expression to a DataFrame library expression
    (bodo.pandas.plan.Expression).
    """
    operator = java_call.getOperator()
    operator_class_name = operator.getClass().getSimpleName()

    if (
        operator_class_name == "SqlMonotonicBinaryOperator"
        and len(java_call.getOperands()) == 2
    ):
        operands = java_call.getOperands()
        left = java_expr_to_python_expr(operands[0], input_plan)
        right = java_expr_to_python_expr(operands[1], input_plan)
        kind = operator.getKind()
        SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind

        if kind.equals(SqlKind.PLUS):
            # TODO: support all BodoSQL data types in backend (including date/time)
            # TODO: upcast output to avoid overflow?
            expr = ArithOpExpression(left.empty_data, left, right, "__add__")
            return expr

    raise NotImplementedError(f"Call operator {operator_class_name} not supported yet")


def java_join_to_python_join(ctx, java_join):
    """Convert a BodoSQL Java join plan to a Python join plan."""
    from bodo.ext import plan_optimizer

    join_info = java_join.analyzeCondition()

    # TODO: support non-equi joins
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

    # TODO: support broadcast join flag
    planComparisonJoin = LogicalComparisonJoin(
        empty_join_out,
        left_plan,
        right_plan,
        join_type,
        key_indices,
    )
    return planComparisonJoin
