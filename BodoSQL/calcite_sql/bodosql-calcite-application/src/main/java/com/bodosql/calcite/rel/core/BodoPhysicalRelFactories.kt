package com.bodosql.calcite.rel.core

import com.bodosql.calcite.adapter.bodo.BodoPhysicalAggregate
import com.bodosql.calcite.adapter.bodo.BodoPhysicalFilter
import com.bodosql.calcite.adapter.bodo.BodoPhysicalIntersect
import com.bodosql.calcite.adapter.bodo.BodoPhysicalJoin
import com.bodosql.calcite.adapter.bodo.BodoPhysicalMinus
import com.bodosql.calcite.adapter.bodo.BodoPhysicalProject
import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.adapter.bodo.BodoPhysicalSort
import com.bodosql.calcite.adapter.bodo.BodoPhysicalUnion
import com.bodosql.calcite.adapter.bodo.BodoPhysicalValues
import com.bodosql.calcite.adapter.iceberg.IcebergFilter
import com.bodosql.calcite.adapter.iceberg.IcebergProject
import com.bodosql.calcite.adapter.iceberg.IcebergRel
import com.bodosql.calcite.adapter.iceberg.IcebergSort
import com.bodosql.calcite.adapter.pandas.PandasProject
import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.adapter.snowflake.SnowflakeAggregate
import com.bodosql.calcite.adapter.snowflake.SnowflakeFilter
import com.bodosql.calcite.adapter.snowflake.SnowflakeProject
import com.bodosql.calcite.adapter.snowflake.SnowflakeRel
import com.bodosql.calcite.adapter.snowflake.SnowflakeSort
import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.tools.BodoRelBuilder
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.Context
import org.apache.calcite.plan.Contexts
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.hep.HepRelVertex
import org.apache.calcite.rel.RelCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.core.CorrelationId
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.core.RelFactories
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.tools.RelBuilderFactory
import org.apache.calcite.util.ImmutableBitSet

object BodoPhysicalRelFactories {
    @JvmField
    val BODO_PHYSICAL_PROJECT_FACTORY: RelFactories.ProjectFactory = RelFactories.ProjectFactory(::createProject)

    @JvmField
    val BODO_PHYSICAL_FILTER_FACTORY: RelFactories.FilterFactory = RelFactories.FilterFactory(::createFilter)

    @JvmField
    val BODO_PHYSICAL_JOIN_FACTORY: RelFactories.JoinFactory = RelFactories.JoinFactory(::createJoin)

    @JvmField
    val BODO_PHYSICAL_SET_OP_FACTORY: RelFactories.SetOpFactory = RelFactories.SetOpFactory(::createSetOp)

    @JvmField
    val BODO_PHYSICAL_AGGREGATE_FACTORY: RelFactories.AggregateFactory = RelFactories.AggregateFactory(::createAggregate)

    @JvmField
    val BODO_PHYSICAL_SORT_FACTORY: RelFactories.SortFactory = RelFactories.SortFactory(::createSort)

    @JvmField
    val BODO_PHYSICAL_VALUES_FACTORY: RelFactories.ValuesFactory = RelFactories.ValuesFactory(::createValues)

    @JvmField
    val DEFAULT_CONTEXT: Context =
        Contexts.of(
            BODO_PHYSICAL_PROJECT_FACTORY,
            BODO_PHYSICAL_FILTER_FACTORY,
            BODO_PHYSICAL_JOIN_FACTORY,
            BODO_PHYSICAL_SET_OP_FACTORY,
            BODO_PHYSICAL_AGGREGATE_FACTORY,
            BODO_PHYSICAL_SORT_FACTORY,
            BODO_PHYSICAL_VALUES_FACTORY,
        )

    @JvmField
    val BODO_PHYSICAL_BUILDER: RelBuilderFactory = BodoRelBuilder.proto(DEFAULT_CONTEXT)

    private fun createProject(
        input: RelNode,
        hints: List<RelHint>,
        childExprs: List<RexNode>,
        fieldNames: List<String?>?,
        variablesSet: Set<CorrelationId>,
    ): RelNode {
        assert(input.convention != null) { "Internal Error in Bodo Physical Builder: Input does not have any convention" }
        if (variablesSet.isNotEmpty()) {
            throw UnsupportedOperationException("Correlation variables are not supported")
        }

        if (input is HepRelVertex) {
            return createProject(input.stripped(), hints, childExprs, fieldNames, variablesSet)
        }

        val retVal =
            if (input.convention == BodoPhysicalRel.CONVENTION) {
                val output = BodoPhysicalProject.create(input, childExprs, fieldNames)
                // Ensure we have a streaming trait updated.
                val inputStreamingTrait = input.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)
                if (inputStreamingTrait != null) {
                    output.copy(output.traitSet.replace(output.expectedOutputBatchingProperty(inputStreamingTrait)), output.inputs)
                } else {
                    output
                }
            } else if (input.convention == SnowflakeRel.CONVENTION) {
                SnowflakeProject.create(
                    input.cluster,
                    input.traitSet,
                    input,
                    childExprs,
                    fieldNames,
                    (input as SnowflakeRel).getCatalogTable(),
                )
            } else if (input.convention == IcebergRel.CONVENTION) {
                IcebergProject.create(
                    input.cluster,
                    input.traitSet,
                    input,
                    childExprs,
                    fieldNames,
                    (input as IcebergRel).getCatalogTable(),
                )
            } else if (input.convention == PandasRel.CONVENTION) {
                val output =
                    PandasProject.create(
                        input.cluster,
                        input.traitSet,
                        input,
                        childExprs,
                        fieldNames,
                    )
                // Ensure we have a streaming trait updated.
                val inputStreamingTrait = input.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)
                if (inputStreamingTrait != null) {
                    output.copy(output.traitSet.replace(output.expectedOutputBatchingProperty(inputStreamingTrait)), output.inputs)
                } else {
                    output
                }
            } else {
                throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder: Unknown convention: " + input.convention?.name)
            }

        return retVal
    }

    private fun createFilter(
        input: RelNode,
        condition: RexNode,
        variablesSet: Set<CorrelationId>,
    ): RelNode {
        assert(input.convention != null) { "Internal Error in Bodo Physical Builder: Input does not have any convention" }
        if (variablesSet.isNotEmpty()) {
            throw UnsupportedOperationException("Correlation variables are not supported")
        }

        if (input is HepRelVertex) {
            return createFilter(input.stripped(), condition, variablesSet)
        }

        val retVal =
            if (input.convention == BodoPhysicalRel.CONVENTION) {
                val output = BodoPhysicalFilter.create(input.cluster, input, condition)
                // Ensure we have a streaming trait updated.
                val inputStreamingTrait = input.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)
                if (inputStreamingTrait != null) {
                    output.copy(output.traitSet.replace(output.expectedOutputBatchingProperty(inputStreamingTrait)), output.inputs)
                } else {
                    output
                }
            } else if (input.convention == SnowflakeRel.CONVENTION) {
                assert(
                    input is SnowflakeRel,
                ) { "Internal Error in Bodo Physical Builder's createFilter: input is not a SnowflakeRel. Input: $input" }
                SnowflakeFilter.create(input.cluster, input.traitSet, input, condition, (input as SnowflakeRel).getCatalogTable())
            } else if (input.convention == IcebergRel.CONVENTION) {
                assert(
                    input is IcebergRel,
                ) { "Internal Error in Bodo Physical Builder's createFilter: input is not a IcebergRel. Input: $input" }
                IcebergFilter.create(input.cluster, input.traitSet, input, condition, (input as IcebergRel).getCatalogTable())
            } else {
                throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder: Unknown convention: " + input.convention?.name)
            }

        return retVal
    }

    private fun createJoin(
        left: RelNode,
        right: RelNode,
        hints: List<RelHint>,
        condition: RexNode,
        variablesSet: Set<CorrelationId>,
        joinType: JoinRelType,
        semiJoinDone: Boolean,
    ): RelNode {
        assert(left.convention != null && right.convention != null) {
            "Internal Error in Bodo Physical Builder: Input does not have any convention"
        }
        assert(left.convention == right.convention) {
            "Internal Error in Bodo Physical Builder's createJoin: Left and Right input do not have the same convention"
        }

        if (semiJoinDone) {
            throw UnsupportedOperationException("Semi-join operation is not supported")
        } else if (variablesSet.isNotEmpty()) {
            throw UnsupportedOperationException("Correlation variables are not supported")
        }

        val inputConvention = left.convention

        val retVal =
            if (inputConvention == BodoPhysicalRel.CONVENTION) {
                val output = BodoPhysicalJoin.create(left, right, hints, condition, joinType)
                // Ensure we have a streaming trait updated.
                val inputStreamingTrait = left.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)
                if (inputStreamingTrait != null) {
                    output.copy(output.traitSet.replace(output.expectedOutputBatchingProperty(inputStreamingTrait)), output.inputs)
                } else {
                    output
                }
            } else if (inputConvention == SnowflakeRel.CONVENTION) {
                throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder's createJoin: unhandled Snowflake operation")
            } else if (inputConvention == IcebergRel.CONVENTION) {
                throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder's createJoin: unhandled Iceberg operation")
            } else {
                throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder: Unknown convention: " + inputConvention?.name)
            }

        return retVal
    }

    private fun createSetOp(
        kind: SqlKind,
        inputs: List<RelNode>,
        all: Boolean,
    ): RelNode {
        assert(inputs.isNotEmpty()) { "Internal Error in Bodo Physical Builder: inputs to SetOp is empty" }
        val firstInput = inputs[0]
        val inputConvention = firstInput.convention
        val cluster = firstInput.cluster
        assert(inputConvention != null) { "Internal Error in Bodo Physical Builder: Input does not have any convention" }
        assert(
            inputs.all { input ->
                input.convention == inputConvention
            },
        ) { "Internal Error in Bodo Physical Builder: inputs to SetOp have differing convention" }

        val retVal =
            if (inputConvention == BodoPhysicalRel.CONVENTION) {
                val output = createBodoPhysicalSetOp(cluster, kind, inputs, all)
                // Ensure we have a streaming trait updated.
                val inputStreamingTrait = firstInput.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)
                if (inputStreamingTrait != null) {
                    output.copy(output.traitSet.replace(output.expectedOutputBatchingProperty(inputStreamingTrait)), output.inputs)
                } else {
                    output
                }
            } else if (inputConvention == SnowflakeRel.CONVENTION) {
                throw BodoSQLCodegenException(
                    "Internal Error in Bodo Physical Builder's createPandasSetOp: unhandled snowflake set operation: " + kind.name,
                )
            } else if (inputConvention == IcebergRel.CONVENTION) {
                throw BodoSQLCodegenException(
                    "Internal Error in Bodo Physical Builder's createPandasSetOp: unhandled Iceberg set operation: " + kind.name,
                )
            } else {
                throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder: Unknown convention: " + inputConvention?.name)
            }

        return retVal
    }

    private fun createBodoPhysicalSetOp(
        cluster: RelOptCluster,
        kind: SqlKind,
        inputs: List<RelNode>,
        all: Boolean,
    ): BodoPhysicalRel =
        when (kind) {
            SqlKind.UNION -> BodoPhysicalUnion.create(cluster, inputs, all)
            SqlKind.INTERSECT -> BodoPhysicalIntersect.create(cluster, inputs, all)
            SqlKind.MINUS -> BodoPhysicalMinus.create(cluster, inputs, all)
            else -> throw BodoSQLCodegenException(
                "Internal Error in Bodo Physical Builder's createBodoPhysicalSetOp: unhandled Bodo Physical set operation: " + kind.name,
            )
        }

    private fun createAggregate(
        input: RelNode,
        hints: List<RelHint>,
        groupSet: ImmutableBitSet,
        groupSets: ImmutableList<ImmutableBitSet>,
        aggCalls: List<AggregateCall>,
    ): RelNode {
        val inputConvention = input.convention
        assert(inputConvention != null) { "Internal Error in Bodo Physical Builder: Input does not have any convention" }

        if (input is HepRelVertex) {
            return createAggregate(input.stripped(), hints, groupSet, groupSets, aggCalls)
        }

        val retVal =
            if (inputConvention == BodoPhysicalRel.CONVENTION) {
                val output = BodoPhysicalAggregate.create(input.cluster, input, groupSet, groupSets, aggCalls)
                // Ensure we have a streaming trait updated.
                val inputStreamingTrait = input.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)
                if (inputStreamingTrait != null) {
                    output.copy(output.traitSet.replace(output.expectedOutputBatchingProperty(inputStreamingTrait)), output.inputs)
                } else {
                    output
                }
            } else if (inputConvention == SnowflakeRel.CONVENTION) {
                SnowflakeAggregate.create(
                    input.cluster,
                    input.traitSet,
                    input,
                    groupSet,
                    groupSets,
                    aggCalls,
                    (input as SnowflakeRel).getCatalogTable(),
                )
            } else if (inputConvention == IcebergRel.CONVENTION) {
                throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder's createAggregate: unhandled Iceberg operation")
            } else {
                throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder: Unknown convention: " + inputConvention?.name)
            }

        return retVal
    }

    private fun createSort(
        input: RelNode,
        collation: RelCollation,
        offset: RexNode?,
        fetch: RexNode?,
    ): RelNode {
        val inputConvention = input.convention
        assert(inputConvention != null) { "Internal Error in Bodo Physical Builder: Input does not have any convention" }

        if (input is HepRelVertex) {
            return createSort(input.stripped(), collation, offset, fetch)
        }

        val retVal =
            if (inputConvention == BodoPhysicalRel.CONVENTION) {
                val output = BodoPhysicalSort.create(input, collation, offset, fetch)
                // Ensure we have a streaming trait updated.
                val inputStreamingTrait = input.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)
                if (inputStreamingTrait != null) {
                    output.copy(output.traitSet.replace(output.expectedOutputBatchingProperty(inputStreamingTrait)), output.inputs)
                } else {
                    output
                }
            } else if (inputConvention == SnowflakeRel.CONVENTION) {
                SnowflakeSort.create(
                    input.cluster,
                    input.traitSet,
                    input,
                    collation,
                    offset,
                    fetch,
                    (input as SnowflakeRel).getCatalogTable(),
                )
            } else if (inputConvention == IcebergRel.CONVENTION) {
                IcebergSort.create(input.cluster, input.traitSet, input, collation, offset, fetch, (input as IcebergRel).getCatalogTable())
            } else {
                throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder: Unknown convention: " + inputConvention?.name)
            }

        return retVal
    }

    private fun createValues(
        cluster: RelOptCluster,
        rowType: RelDataType,
        tuples: List<ImmutableList<RexLiteral>>,
    ): RelNode {
        val immutableTuples = ImmutableList.copyOf(tuples)
        val values = BodoPhysicalValues.create(cluster, cluster.traitSet().replace(BodoPhysicalRel.CONVENTION), rowType, immutableTuples)
        // Values doesn't have an input, but this is still valid for producing a null check.
        // BodoPhysicalValues will ignore in the input trait being passed in.
        val inputStreamingTrait = cluster.traitSet().getTrait(BatchingPropertyTraitDef.INSTANCE)
        return if (inputStreamingTrait != null) {
            values.copy(values.traitSet.replace(values.expectedOutputBatchingProperty(inputStreamingTrait)), values.inputs)
        } else {
            values
        }
    }
}
