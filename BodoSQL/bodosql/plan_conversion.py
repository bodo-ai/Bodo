from __future__ import annotations

import operator
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa

import bodo
import bodo.pandas as bd
import bodosql
from bodo.pandas.plan import (
    AggregateExpression,
    ArithOpExpression,
    ComparisonOpExpression,
    ConjunctionOpExpression,
    ConstantExpression,
    LogicalAggregate,
    LogicalComparisonJoin,
    LogicalFilter,
    LogicalJoinFilter,
    LogicalOrder,
    LogicalProjection,
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

    if java_class_name == "BodoPhysicalRuntimeJoinFilter":
        return java_rtjf_to_python_rtjf(ctx, java_plan)

    if java_class_name == "BodoPhysicalFilter":
        return java_filter_to_python_filter(ctx, java_plan)

    if java_class_name == "BodoPhysicalAggregate" and not java_plan.usesGroupingSets():
        # TODO: support grouping sets
        return java_agg_to_python_agg(ctx, java_plan)

    if java_class_name == "BodoPhysicalSort":
        return java_sort_to_python_sort(ctx, java_plan)

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

    if operator_class_name in ("SqlMonotonicBinaryOperator", "SqlBinaryOperator"):
        operands = java_call.getOperands()
        # Calcite may add more than 2 operand for the same binary operator
        op_exprs = [java_expr_to_python_expr(o, input_plan) for o in operands]
        kind = op.getKind()
        return java_binop_to_python_expr(kind, op_exprs)

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
            return java_expr_to_python_expr(operand, input_plan)

        if operand_type.getSqlTypeName().equals(
            SqlTypeName.VARCHAR
        ) and target_type.getSqlTypeName().equals(SqlTypeName.VARCHAR):
            # No-op cast of VARCHAR (could be different lengths but sometimes equal
            # which seems like a Calcite gap)
            return java_expr_to_python_expr(operand, input_plan)

    if (
        operator_class_name == "SqlPostfixOperator"
        and len(java_call.getOperands()) == 1
    ):
        operands = java_call.getOperands()
        input = java_expr_to_python_expr(operands[0], input_plan)
        kind = op.getKind()
        SqlKind = gateway.jvm.org.apache.calcite.sql.SqlKind

        if kind.equals(SqlKind.IS_NOT_NULL):
            bool_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.bool_()))
            return UnaryOpExpression(bool_empty_data, input, "notnull")

    raise NotImplementedError(
        f"Call operator {operator_class_name} not supported yet: "
        + java_call.toString()
    )


def java_binop_to_python_expr(kind, op_exprs):
    """Convert a BodoSQL Java binary operator call to a DataFrame library expression."""

    left = op_exprs[0]

    # Calcite may add more than 2 operand for the same binary operator
    if len(op_exprs) > 2:
        right = java_binop_to_python_expr(kind, op_exprs[1:])
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


def java_join_to_python_join(ctx, java_join):
    """Convert a BodoSQL Java join plan to a Python join plan."""
    from bodo.ext import plan_optimizer

    ctx.join_filter_info[java_join.getJoinFilterID()] = (
        java_join.getOriginalJoinFilterKeyLocations()
    )

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
        java_join.getJoinFilterID(),
    )
    return planComparisonJoin


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
    condition = java_expr_to_python_expr(java_filter.getCondition(), input_plan)
    return LogicalFilter(input_plan.empty_data, input_plan, condition)


def java_literal_to_python_literal(java_literal, input_plan):
    """Convert a BodoSQL Java literal expression to a DataFrame library constant"""
    SqlTypeName = gateway.jvm.org.apache.calcite.sql.type.SqlTypeName
    lit_type_name = java_literal.getTypeName()
    lit_type = java_literal.getType()

    # TODO[BSE-5156]: support all Calcite literal types

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

    if lit_type_name.equals(SqlTypeName.DOUBLE):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
        return ConstantExpression(dummy_empty_data, input_plan, java_literal.getValue())

    if lit_type_name.equals(SqlTypeName.CHAR):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.large_string()))
        return ConstantExpression(
            dummy_empty_data, input_plan, java_literal.getValue2()
        )

    if lit_type_name.equals(SqlTypeName.DATE):
        dummy_empty_data = pd.Series(dtype=pd.ArrowDtype(pa.date32()))
        # getValue2() returns an integer representing days since epoch
        val = pa.scalar(java_literal.getValue2(), pa.date32())
        return ConstantExpression(dummy_empty_data, input_plan, val)

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

    if len(keys) == 0:
        raise NotImplementedError("Aggregations without group by not supported yet")

    input_plan = java_plan_to_python_plan(ctx, java_plan.getInput())

    exprs = []
    out_types = [input_plan.pa_schema.field(k).type for k in keys]
    for func in java_plan.getAggCallList():
        if func.hasFilter():
            raise NotImplementedError("Filtered aggregations are not supported yet")
        func_name = _agg_to_func_name(func)
        arg_cols = list(func.getArgList())
        if func_name == "size":
            out_type = pa.int64()
        else:
            assert len(arg_cols) == 1, "Only single-argument aggregations are supported"
            in_type = input_plan.pa_schema.field(arg_cols[0]).type
            out_type = _get_agg_output_type(
                GroupbyAggFunc("dummy", func_name), in_type, "dummy"
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

    # TODO[BSE-5163]: support SUM0 initialization properly
    if kind.equals(SqlKind.SUM) or kind.equals(SqlKind.SUM0):
        return "sum"

    if kind.equals(SqlKind.COUNT) and len(func.getArgList()) == 0:
        return "size"

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

    raise NotImplementedError(
        f"Call operator {operator_class_name} not supported yet: "
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
