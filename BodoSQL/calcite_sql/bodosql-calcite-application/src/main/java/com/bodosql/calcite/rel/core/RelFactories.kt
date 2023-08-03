package com.bodosql.calcite.rel.core

import com.bodosql.calcite.rel.logical.*
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.Context
import org.apache.calcite.plan.Contexts
import org.apache.calcite.rel.RelCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.core.CorrelationId
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.core.RelFactories.*
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.tools.RelBuilder
import org.apache.calcite.tools.RelBuilderFactory
import org.apache.calcite.util.ImmutableBitSet
import org.checkerframework.checker.nullness.qual.Nullable

object RelFactories {
    @JvmField
    val PROJECT_FACTORY: ProjectFactory = ProjectFactory(::createProject)

    @JvmField
    val FILTER_FACTORY: FilterFactory = FilterFactory(::createFilter)

    @JvmField
    val JOIN_FACTORY: JoinFactory = JoinFactory(::createJoin)

    @JvmField
    val SET_OP_FACTORY: SetOpFactory = SetOpFactory(::createSetOp)

    @JvmField
    val AGGREGATE_FACTORY: AggregateFactory = AggregateFactory(::createAggregate)

    @JvmField
    val SORT_FACTORY: SortFactory = SortFactory(::createSort)

    @JvmField
    val DEFAULT_CONTEXT: Context = Contexts.of(
        PROJECT_FACTORY,
        FILTER_FACTORY,
        JOIN_FACTORY,
        SET_OP_FACTORY,
        AGGREGATE_FACTORY,
        SORT_FACTORY,
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

    private fun createSetOp(kind: SqlKind, inputs: List<RelNode>, all: Boolean): RelNode {
        return when (kind) {
            SqlKind.UNION -> BodoLogicalUnion.create(inputs, all)
            else -> DEFAULT_SET_OP_FACTORY.createSetOp(kind, inputs, all);
        }
    }

    private fun createAggregate(input: RelNode, hints: List<RelHint>, groupSet: ImmutableBitSet,
                                groupSets: ImmutableList<ImmutableBitSet>, aggCalls: List<AggregateCall>): RelNode {
        return BodoLogicalAggregate.create(input, hints, groupSet, groupSets, aggCalls)
    }

    private fun createSort(input: RelNode, collation: RelCollation, offset: RexNode?, fetch: RexNode?) : RelNode {
        return BodoLogicalSort.create(input, collation, offset, fetch)
    }
}
