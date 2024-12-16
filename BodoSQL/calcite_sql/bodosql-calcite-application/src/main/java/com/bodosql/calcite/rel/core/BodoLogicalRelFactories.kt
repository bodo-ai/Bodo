package com.bodosql.calcite.rel.core

import com.bodosql.calcite.rel.logical.BodoLogicalAggregate
import com.bodosql.calcite.rel.logical.BodoLogicalFilter
import com.bodosql.calcite.rel.logical.BodoLogicalJoin
import com.bodosql.calcite.rel.logical.BodoLogicalProject
import com.bodosql.calcite.rel.logical.BodoLogicalSort
import com.bodosql.calcite.rel.logical.BodoLogicalUnion
import com.bodosql.calcite.tools.BodoRelBuilder
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.Context
import org.apache.calcite.plan.Contexts
import org.apache.calcite.rel.RelCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.core.CorrelationId
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.core.RelFactories
import org.apache.calcite.rel.core.RelFactories.DEFAULT_SET_OP_FACTORY
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.tools.RelBuilderFactory
import org.apache.calcite.util.ImmutableBitSet

object BodoLogicalRelFactories {
    @JvmField
    val BODO_LOGICAL_PROJECT_FACTORY: RelFactories.ProjectFactory = RelFactories.ProjectFactory(::createProject)

    @JvmField
    val BODO_LOGICAL_FILTER_FACTORY: RelFactories.FilterFactory = RelFactories.FilterFactory(::createFilter)

    @JvmField
    val BODO_LOGICAL_JOIN_FACTORY: RelFactories.JoinFactory = RelFactories.JoinFactory(::createJoin)

    @JvmField
    val BODO_LOGICAL_SET_OP_FACTORY: RelFactories.SetOpFactory = RelFactories.SetOpFactory(::createSetOp)

    @JvmField
    val BODO_LOGICAL_AGGREGATE_FACTORY: RelFactories.AggregateFactory = RelFactories.AggregateFactory(::createAggregate)

    @JvmField
    val BODO_LOGICAL_SORT_FACTORY: RelFactories.SortFactory = RelFactories.SortFactory(::createSort)

    @JvmField
    val DEFAULT_CONTEXT: Context =
        Contexts.of(
            BODO_LOGICAL_PROJECT_FACTORY,
            BODO_LOGICAL_FILTER_FACTORY,
            BODO_LOGICAL_JOIN_FACTORY,
            BODO_LOGICAL_SET_OP_FACTORY,
            BODO_LOGICAL_AGGREGATE_FACTORY,
            BODO_LOGICAL_SORT_FACTORY,
        )

    @JvmField
    val BODO_LOGICAL_BUILDER: RelBuilderFactory = BodoRelBuilder.proto(DEFAULT_CONTEXT)

    private fun createProject(
        input: RelNode,
        hints: List<RelHint>,
        childExprs: List<RexNode>,
        fieldNames: List<String?>?,
        variablesSet: Set<CorrelationId>,
    ): RelNode =
        if (variablesSet.isNotEmpty()) {
            throw UnsupportedOperationException("Correlation variables are not supported")
        } else {
            BodoLogicalProject.create(input, hints, childExprs, fieldNames)
        }

    private fun createFilter(
        input: RelNode,
        condition: RexNode,
        variablesSet: Set<CorrelationId>,
    ): RelNode =
        if (variablesSet.isNotEmpty()) {
            throw UnsupportedOperationException("Correlation variables are not supported")
        } else {
            BodoLogicalFilter.create(input, condition)
        }

    private fun createJoin(
        left: RelNode,
        right: RelNode,
        hints: List<RelHint>,
        condition: RexNode,
        variablesSet: Set<CorrelationId>,
        joinType: JoinRelType,
        semiJoinDone: Boolean,
    ): RelNode =
        if (semiJoinDone) {
            throw UnsupportedOperationException("Semi-join operation is not supported")
        } else if (variablesSet.isNotEmpty()) {
            throw UnsupportedOperationException("Correlation variables are not supported")
        } else {
            BodoLogicalJoin.create(left, right, hints, condition, joinType)
        }

    private fun createSetOp(
        kind: SqlKind,
        inputs: List<RelNode>,
        all: Boolean,
    ): RelNode =
        when (kind) {
            SqlKind.UNION -> BodoLogicalUnion.create(inputs, all)
            else -> DEFAULT_SET_OP_FACTORY.createSetOp(kind, inputs, all)
        }

    private fun createAggregate(
        input: RelNode,
        hints: List<RelHint>,
        groupSet: ImmutableBitSet,
        groupSets: ImmutableList<ImmutableBitSet>,
        aggCalls: List<AggregateCall>,
    ): RelNode = BodoLogicalAggregate.create(input, hints, groupSet, groupSets, aggCalls)

    private fun createSort(
        input: RelNode,
        collation: RelCollation,
        offset: RexNode?,
        fetch: RexNode?,
    ): RelNode = BodoLogicalSort.create(input, collation, offset, fetch)
}
