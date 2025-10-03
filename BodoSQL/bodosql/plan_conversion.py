from __future__ import annotations

import pyarrow as pa

import bodo
from bodo.pandas.plan import LogicalProjection, arrow_to_empty_df, make_col_ref_exprs
from bodosql.imported_java_classes import JavaEntryPoint


def java_plan_to_python_plan(ctx, java_plan):
    """Convert a BodoSQL Java plan (RelNode) to a DataFrame library plan
    (bodo.pandas.plan.LazyPlan) for execution in the C++ runtime backend.
    """
    java_class_name = java_plan.getClass().getSimpleName()

    if java_class_name == "PandasToBodoPhysicalConverter":
        # PandasToBodoPhysicalConverter is a no-op
        input = java_plan.getInput()
        return java_plan_to_python_plan(ctx, input)

    if java_class_name == "PandasTableScan":
        # TODO: support other table types and check table details
        table_name = JavaEntryPoint.getLocalTableName(java_plan)
        df = ctx.tables[table_name]
        return bodo.pandas.from_pandas(df)._plan

    if java_class_name == "PandasProject":
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

    raise NotImplementedError(f"Plan node {java_class_name} not supported yet")


def java_expr_to_python_expr(java_expr, input_plan):
    """Convert a BodoSQL Java expression to a DataFrame library expression
    (bodo.pandas.plan.Expression).
    """
    java_class_name = java_expr.getClass().getSimpleName()

    if java_class_name == "RexInputRef":
        col_index = java_expr.getIndex()
        return make_col_ref_exprs([col_index], input_plan)[0]

    raise NotImplementedError(f"Expression {java_class_name} not supported yet")
