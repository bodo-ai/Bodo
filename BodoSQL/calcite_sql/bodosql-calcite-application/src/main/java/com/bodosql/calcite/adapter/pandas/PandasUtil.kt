package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.rel.logical.BodoLogicalProject
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelRoot
import org.apache.calcite.sql.SqlKind
import java.lang.AssertionError

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
    } else {
        // Either the name or some change has been made
        // to the type. Insert a PandasProject to fix everything.
        if (!this.rel.traitSet.contains(PandasRel.CONVENTION)) {
            throw AssertionError("input rel must have pandas convention")
        }
        val rexBuilder = this.rel.cluster.rexBuilder
        val projects = fields.map { (i, _) ->
            rexBuilder.makeInputRef(this.rel, i)
        }
        PandasProject(this.rel.cluster, this.rel.traitSet, this.rel, projects, validatedRowType)
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
    } else {
        val rexBuilder = this.rel.cluster.rexBuilder
        val projects = fields.map { (i, _) ->
            rexBuilder.makeInputRef(this.rel, i)
        }
        BodoLogicalProject.create(this.rel, listOf(), projects, validatedRowType)
    }
