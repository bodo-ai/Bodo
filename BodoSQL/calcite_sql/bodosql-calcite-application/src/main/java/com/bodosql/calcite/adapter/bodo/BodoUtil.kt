package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.rel.logical.BodoLogicalProject
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelRoot
import org.apache.calcite.rel.core.TableCreate
import org.apache.calcite.rel.core.TableModify
import org.apache.calcite.rel.logical.LogicalProject
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.SqlKind

/**
 * Determine if the fields of a DML operation's input
 * match the expected fields.
 */
private fun insertFieldMatch(rel: TableModify): Boolean {
    val input = rel.getInput(0)
    val inputRowType = input.rowType
    val targetRowType = rel.table!!.rowType
    return targetRowType.fieldNames == inputRowType.fieldNames
}

/**
 * Create a Bodo Physical Project Node at the top of the tree for the given input.
 * @param input The input node to project
 * @return A BodoPhysicalProject node that projects the input node
 */
private fun RelRoot.createBodoPhysicalProjectNode(input: RelNode): RelNode {
    if (!input.traitSet.contains(BodoPhysicalRel.CONVENTION)) {
        throw AssertionError("input rel must have Bodo convention")
    }
    val rexBuilder = input.cluster.rexBuilder
    val projects =
        fields.map { (i, _) ->
            rexBuilder.makeInputRef(input, i)
        }
    // Note: We must pass the trait set to ensure we forward streaming information.
    return BodoPhysicalProject.create(input.cluster, input.traitSet, input, projects, validatedRowType)
}

/**
 * Create the projection for an insert which may not match the fields because
 * it uses the actual table type information. Note: We assume insert into already
 * enforces column ordering in SqlToRelConverter.convertColumnList.
 * @param input The input that needs its columns renamed.
 * @param targetType The target type that the input should match.
 * @return A new projection with the correct names.
 */
private fun createInsertBodoPhysicalProjectNode(
    input: RelNode,
    targetType: RelDataType,
): RelNode {
    if (!input.traitSet.contains(BodoPhysicalRel.CONVENTION)) {
        throw AssertionError("input rel must have Bodo convention")
    }
    val rexBuilder = input.cluster.rexBuilder
    // Note: We just use the names because is not required that type nullability match
    // for insert.
    val projects =
        List(targetType.fieldNames.size) { i ->
            val inputRef = rexBuilder.makeInputRef(input, i)
            inputRef
        }
    // Note: We must pass the trait set to ensure we forward streaming information.
    return BodoPhysicalProject.create(input.cluster, input.traitSet, input, projects, targetType.fieldNames)
}

/**
 * Transforms a [RelRoot] to a [RelNode] that can be used for code generation.
 *
 * See [RelRoot] for the rationalization of why that node is needed. This method
 * does the equivalent of [RelRoot.project] but also takes into account the
 * names and uses a [BodoPhysicalProject] instead of a [org.apache.calcite.rel.logical.LogicalProject]
 * for the generated projection.
 *
 * If no projection is needed to satisfy the output, this method returns the original
 * [RelNode].
 *
 * @return [RelNode] that corresponds to the final node of this [RelRoot]
 */
fun RelRoot.bodoPhysicalProject(): RelNode =
    if (kind.belongsTo(SqlKind.DML)) {
        // DML operations are special in that the validated node type
        // is for the inner SELECT and is not related to the output.
        if (this.kind != SqlKind.INSERT || insertFieldMatch(this.rel as TableModify)) {
            this.rel
        } else {
            val tableModify = this.rel as TableModify
            val newInput = createInsertBodoPhysicalProjectNode(tableModify.getInput(0), tableModify.table!!.getRowType())
            this.rel.copy(this.rel.traitSet, listOf(newInput))
        }
    } else if (isRefTrivial && isNameTrivial) {
        // No transformation is needed.
        this.rel
    } else if (kind == SqlKind.CREATE_TABLE) {
        // Create table is special because the validated type is for the actual
        // select. We need to check that the names and columns match the expectations
        // for the RelRoot.
        // TODO: Add other DDL operations?
        if (this.rel !is TableCreate) {
            throw AssertionError("Input create sqlkind doesn't match a supported node")
        }
        // Insert the projection before the CREATE_TABLE
        val newInput = createBodoPhysicalProjectNode(this.rel.getInput(0))
        this.rel.copy(this.rel.traitSet, listOf(newInput))
    } else {
        // Either the name or some change has been made
        createBodoPhysicalProjectNode(this.rel)
    }

/**
 * Create the projection for an insert which may not match the fields because
 * it uses the actual table type information. Note: We assume insert into already
 * enforces column ordering in SqlToRelConverter.convertColumnList.
 * @param input The input that needs its columns renamed.
 * @param targetType The target type that the input should match.
 * @return A new projection with the correct names.
 */
private fun createInsertLogicalProjectNode(
    input: RelNode,
    targetType: RelDataType,
): RelNode {
    val rexBuilder = input.cluster.rexBuilder
    // Note: We just use the names because is not required that type nullability match
    // for insert.
    val projects =
        List(targetType.fieldNames.size) { i ->
            val inputRef = rexBuilder.makeInputRef(input, i)
            inputRef
        }
    // Note: We must pass the trait set to ensure we forward streaming information.
    return BodoLogicalProject.create(input, listOf(), projects, targetType.fieldNames)
}

/**
 * Create a BodoLogicalProject Node at the top of the tree for the given input.
 * @param input The input node to project
 * @return A BodoLogicalProject node that projects the input node.
 */
private fun RelRoot.createLogicalProjectNode(input: RelNode): RelNode {
    val rexBuilder = input.cluster.rexBuilder
    val projects =
        fields.map { (i, _) ->
            rexBuilder.makeInputRef(input, i)
        }
    return BodoLogicalProject.create(input, listOf(), projects, validatedRowType)
}

/**
 * Create a LogicalProject Node at the top of the tree for the given input.
 * @param input The input node to project
 * @return A BodoLogicalProject node that projects the input node.
 */
private fun RelRoot.createCalciteLogicalProjectNode(input: RelNode): RelNode {
    val rexBuilder = input.cluster.rexBuilder
    val projects =
        fields.map { (i, _) ->
            val inputRef = rexBuilder.makeInputRef(input, i)
            val expectedType = validatedRowType.fieldList[i].type
            // Its possible the literal values may not match the Bodo
            // expected types. This notably occurs when inlining views
            // if we treat Number(38, 0) as BIGINT.
            if (inputRef.type != expectedType) {
                rexBuilder.makeCast(expectedType, inputRef)
            } else {
                inputRef
            }
        }
    return LogicalProject.create(input, listOf(), projects, validatedRowType, setOf())
}

/**
 * Transforms a [RelRoot] to a [RelNode] that can be used for output plan
 * generation. This is not needed for core functionality, but it is included
 * to avoid confusion that could arise from calling RelRoot.project() directly,
 * which would output a LogicalProject instead of a BodoLogicalProject.
 *
 * See [RelRoot] for the rationalization of why that node is needed. This method
 * does the equivalent of [RelRoot.project] but also takes into account the
 * names and uses a [BodoLogicalProject] instead of a [org.apache.calcite.rel.logical.LogicalProject]
 * for the generated projection.
 *
 * If no projection is needed to satisfy the output, this method returns the original
 * [RelNode].
 *
 * @return [RelNode] that corresponds to the final node of this [RelRoot]
 */
fun RelRoot.logicalProject(): RelNode =
    if (kind.belongsTo(SqlKind.DML)) {
        // DML operations are special in that the validated node type
        // is for the inner SELECT and is not related to the output.
        if (this.kind != SqlKind.INSERT || insertFieldMatch(this.rel as TableModify)) {
            this.rel
        } else {
            val tableModify = this.rel as TableModify
            val newInput = createInsertLogicalProjectNode(tableModify.getInput(0), tableModify.table!!.getRowType())
            this.rel.copy(this.rel.traitSet, listOf(newInput))
        }
    } else if (isRefTrivial && isNameTrivial) {
        // No transformation is needed.
        this.rel
    } else if (kind == SqlKind.CREATE_TABLE) {
        // Create table is special because the validated type is for the actual
        // select. We need to check that the names and columns match the expectations
        // for the RelRoot.
        // TODO: Add other DDL operations?
        if (this.rel !is TableCreate) {
            throw AssertionError("Input create sqlkind doesn't match a supported node")
        }
        // Insert the projection before the CREATE_TABLE
        val newInput = createLogicalProjectNode(this.rel.getInput(0))
        this.rel.copy(this.rel.traitSet, listOf(newInput))
    } else {
        createLogicalProjectNode(this.rel)
    }

/**
 * Equivalent to RelRoot.project() for creating Calcite
 * logical nodes, but with the additional difference that
 * it checks for aliases and that it allows for type casting
 * if the generated type doesn't match the validated type.
 */
fun RelRoot.calciteLogicalProject(): RelNode =
    if (kind.belongsTo(SqlKind.DML)) {
        if (this.kind != SqlKind.INSERT || insertFieldMatch(this.rel as TableModify)) {
            this.rel
        } else {
            // Note we don't need types to match exactly for nullability. Also, this code shouldn't
            // be reached most likely.
            val tableModify = this.rel as TableModify
            val newInput = createInsertLogicalProjectNode(tableModify.getInput(0), tableModify.table!!.getRowType())
            this.rel.copy(this.rel.traitSet, listOf(newInput))
        }
    } else if (isRefTrivial && isNameTrivial && this.rel.getRowType() == validatedRowType) {
        // No transformation is needed because types match exactly.
        this.rel
    } else if (kind == SqlKind.CREATE_TABLE) {
        // Create table is special because the validated type is for the actual
        // select. We need to check that the names and columns match the expectations
        // for the RelRoot.
        // TODO: Add other DDL operations?
        if (this.rel !is TableCreate) {
            throw AssertionError("Input create sqlkind doesn't match a supported node")
        }
        // Insert the projection before the CREATE_TABLE
        val newInput = createCalciteLogicalProjectNode(this.rel.getInput(0))
        this.rel.copy(this.rel.traitSet, listOf(newInput))
    } else {
        createCalciteLogicalProjectNode(this.rel)
    }
