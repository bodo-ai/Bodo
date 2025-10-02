from __future__ import annotations

import bodo
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

    raise NotImplementedError(f"Plan node {java_class_name} not supported yet")
