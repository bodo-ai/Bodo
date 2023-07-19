package com.bodosql.calcite.rel.core

import com.bodosql.calcite.rel.logical.BodoLogicalFilter
import com.bodosql.calcite.rel.logical.BodoLogicalJoin
import com.bodosql.calcite.rel.logical.BodoLogicalProject
import org.apache.calcite.plan.Context
import org.apache.calcite.plan.Contexts
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.CorrelationId
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.core.RelFactories.FilterFactory
import org.apache.calcite.rel.core.RelFactories.JoinFactory
import org.apache.calcite.rel.core.RelFactories.ProjectFactory
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rex.RexNode
import org.apache.calcite.tools.RelBuilder
import org.apache.calcite.tools.RelBuilderFactory

object RelFactories {
    @JvmField
    val PROJECT_FACTORY: ProjectFactory = ProjectFactory(::createProject)

    @JvmField
    val FILTER_FACTORY: FilterFactory = FilterFactory(::createFilter)

    @JvmField
    val JOIN_FACTORY: JoinFactory = JoinFactory(::createJoin)

    @JvmField
    val DEFAULT_CONTEXT: Context = Contexts.of(
        PROJECT_FACTORY,
        FILTER_FACTORY,
        JOIN_FACTORY,
    )

    @JvmField
    val LOGICAL_BUILDER: RelBuilderFactory = RelBuilder.proto(DEFAULT_CONTEXT)

    private fun createProject(input: RelNode, hints: List<RelHint>, childExprs: List<RexNode>, fieldNames: List<String?>?): RelNode =
        BodoLogicalProject.create(input, hints, childExprs, fieldNames)

    private fun createFilter(input: RelNode, condition: RexNode, variablesSet: Set<CorrelationId>): RelNode =
        if (variablesSet.isNotEmpty()) {
            throw UnsupportedOperationException("Correlation variables are not supported")
        } else {
            BodoLogicalFilter.create(input, condition)
        }

    private fun createJoin(left: RelNode, right: RelNode, hints: List<RelHint>,
                           condition: RexNode, variablesSet: Set<CorrelationId>,
                           joinType: JoinRelType,
                           semiJoinDone: Boolean): RelNode =
        if (semiJoinDone) {
            throw UnsupportedOperationException("Semi-join operation is not supported")
        } else if (variablesSet.isNotEmpty()) {
            throw UnsupportedOperationException("Correlation variables are not supported")
        } else {
            BodoLogicalJoin.create(left, right, hints, condition, joinType)
        }
}
