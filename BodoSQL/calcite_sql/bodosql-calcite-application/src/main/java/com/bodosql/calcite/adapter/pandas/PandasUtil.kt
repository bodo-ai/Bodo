package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.rel.logical.BodoLogicalProject
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelRoot
import org.apache.calcite.rel.core.TableCreate
import org.apache.calcite.rel.logical.LogicalProject
import org.apache.calcite.sql.SqlKind
import java.lang.AssertionError

/**
 * Create a Pandas Project Node at the top of the tree for the given input.
 */
private fun RelRoot.createPandasProjectNode(input: RelNode): RelNode {
    if (!input.traitSet.contains(PandasRel.CONVENTION)) {
        throw AssertionError("input rel must have pandas convention")
    }
    val rexBuilder = input.cluster.rexBuilder
    val projects =
        fields.map { (i, _) ->
            rexBuilder.makeInputRef(input, i)
        }
    return PandasProject(input.cluster, input.traitSet, input, projects, validatedRowType)
}

/**
 * Transforms a [RelRoot] to a [RelNode] that can be used for code generation.
 *
 * See [RelRoot] for the rationalization of why that node is needed. This method
 * does the equivalent of [RelRoot.project] but also takes into account the
 * names and uses a [PandasProject] instead of a [org.apache.calcite.rel.logical.LogicalProject]
 * for the generated projection.
 *
 * If no projection is needed to satisfy the output, this method returns the original
 * [RelNode].
 *
 * @return [RelNode] that corresponds to the final node of this [RelRoot]
 */
fun RelRoot.pandasProject(): RelNode =
    if (isRefTrivial && (kind.belongsTo(SqlKind.DML) || isNameTrivial)) {
        // No transformation is needed.
        // DML operations are special in that the validated node type
        // is for the inner SELECT and is not related to the output.
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
        val newInput = createPandasProjectNode(this.rel.getInput(0))
        this.rel.copy(this.rel.traitSet, listOf(newInput))
    } else {
        // Either the name or some change has been made
        createPandasProjectNode(this.rel)
    }

/**
 * Create a BodoLogicalProject Node at the top of the tree for the given input.
 */
private fun RelRoot.createLogicalProjectNode(input: RelNode): RelNode {
    val rexBuilder = input.cluster.rexBuilder
    val projects =
        fields.map { (i, _) ->
            rexBuilder.makeInputRef(input, i)
        }
    return BodoLogicalProject.create(input, listOf(), projects, validatedRowType)
}

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
    if (isRefTrivial && (kind.belongsTo(SqlKind.DML) || isNameTrivial)) {
        // No transformation is needed.
        // DML operations are special in that the validated node type
        // is for the inner SELECT and is not related to the output.
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
    if (isRefTrivial && kind.belongsTo(SqlKind.DML)) {
        // No transformation is needed.
        // DML operations are special in that the validated node type
        // is for the inner SELECT and is not related to the output.
        this.rel
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
