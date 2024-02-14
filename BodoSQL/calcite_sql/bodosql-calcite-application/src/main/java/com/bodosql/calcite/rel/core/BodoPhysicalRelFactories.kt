package com.bodosql.calcite.rel.core

import com.bodosql.calcite.adapter.iceberg.IcebergFilter
import com.bodosql.calcite.adapter.iceberg.IcebergProject
import com.bodosql.calcite.adapter.iceberg.IcebergRel
import com.bodosql.calcite.adapter.iceberg.IcebergSort
import com.bodosql.calcite.adapter.pandas.PandasAggregate
import com.bodosql.calcite.adapter.pandas.PandasFilter
import com.bodosql.calcite.adapter.pandas.PandasJoin
import com.bodosql.calcite.adapter.pandas.PandasProject
import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.adapter.pandas.PandasSort
import com.bodosql.calcite.adapter.pandas.PandasTableScan
import com.bodosql.calcite.adapter.pandas.PandasUnion
import com.bodosql.calcite.adapter.pandas.PandasValues
import com.bodosql.calcite.adapter.snowflake.SnowflakeAggregate
import com.bodosql.calcite.adapter.snowflake.SnowflakeFilter
import com.bodosql.calcite.adapter.snowflake.SnowflakeProject
import com.bodosql.calcite.adapter.snowflake.SnowflakeRel
import com.bodosql.calcite.adapter.snowflake.SnowflakeSort
import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan
import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.table.BodoSqlTable
import com.bodosql.calcite.table.SnowflakeCatalogTable
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.Context
import org.apache.calcite.plan.Contexts
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.hep.HepRelVertex
import org.apache.calcite.prepare.RelOptTableImpl
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
import org.apache.calcite.tools.RelBuilder
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
    val BODO_PHYSICAL_TABLE_SCAN_FACTORY: RelFactories.TableScanFactory = RelFactories.TableScanFactory(::createTableScan)

    @JvmField
    val BODO_PHYSICAL_VALUES_FACTORY: RelFactories.ValuesFactory = RelFactories.ValuesFactory(::createValues)

    @JvmField
    val DEFAULT_CONTEXT: Context = Contexts.of(
        BODO_PHYSICAL_PROJECT_FACTORY,
        BODO_PHYSICAL_FILTER_FACTORY,
        BODO_PHYSICAL_JOIN_FACTORY,
        BODO_PHYSICAL_SET_OP_FACTORY,
        BODO_PHYSICAL_AGGREGATE_FACTORY,
        BODO_PHYSICAL_SORT_FACTORY,
        BODO_PHYSICAL_TABLE_SCAN_FACTORY,
        BODO_PHYSICAL_VALUES_FACTORY,
    )

    @JvmField
    val BODO_PHYSICAL_BUILDER: RelBuilderFactory = RelBuilder.proto(DEFAULT_CONTEXT)

    private fun createProject(input: RelNode, hints: List<RelHint>, childExprs: List<RexNode>, fieldNames: List<String?>?, variablesSet: Set<CorrelationId>): RelNode {
        assert(input.convention != null) { "Internal Error in Bodo Physical Builder: Input does not have any convention" }
        if (variablesSet.isNotEmpty()) {
            throw UnsupportedOperationException("Correlation variables are not supported")
        }

        val strippedInput = if (input is HepRelVertex) {
            input.stripped()
        } else {
            input
        }

        val retVal = if (strippedInput.convention == PandasRel.CONVENTION) {
            PandasProject.create(strippedInput, childExprs, fieldNames)
        } else if (strippedInput.convention == SnowflakeRel.CONVENTION) {
            SnowflakeProject.create(
                strippedInput.cluster,
                strippedInput.traitSet,
                strippedInput,
                childExprs,
                fieldNames,
                (strippedInput as SnowflakeRel).getCatalogTable(),
            )
        } else if (strippedInput.convention == IcebergRel.CONVENTION) {
            IcebergProject.create(
                strippedInput.cluster,
                strippedInput.traitSet,
                strippedInput,
                childExprs,
                fieldNames,
                (strippedInput as IcebergRel).getCatalogTable(),
            )
        } else {
            throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder: Unknown convention: " + strippedInput.convention?.name)
        }

        assert(retVal.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE) == strippedInput.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)) { "TODO createProject: fix Batching property" }
        return retVal
    }

    private fun createFilter(input: RelNode, condition: RexNode, variablesSet: Set<CorrelationId>): RelNode {
        assert(input.convention != null) { "Internal Error in Bodo Physical Builder: Input does not have any convention" }
        if (variablesSet.isNotEmpty()) {
            throw UnsupportedOperationException("Correlation variables are not supported")
        }

        val strippedInput = if (input is HepRelVertex) {
            input.stripped()
        } else {
            input
        }

        val retVal = if (strippedInput.convention == PandasRel.CONVENTION) {
            PandasFilter.create(strippedInput.cluster, strippedInput, condition)
        } else if (strippedInput.convention == SnowflakeRel.CONVENTION) {
            SnowflakeFilter.create(strippedInput.cluster, strippedInput.traitSet, strippedInput, condition, (strippedInput as SnowflakeRel).getCatalogTable())
        } else if (input.convention == IcebergRel.CONVENTION) {
            IcebergFilter.create(strippedInput.cluster, strippedInput.traitSet, strippedInput, condition, (strippedInput as IcebergRel).getCatalogTable())
        } else {
            throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder: Unknown convention: " + input.convention?.name)
        }

        assert(retVal.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE) == strippedInput.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)) { "TODO createProject: fix Batching property" }
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
        assert(left.convention != null && right.convention != null) { "Internal Error in Bodo Physical Builder: Input does not have any convention" }
        assert(left.convention == right.convention) { "Internal Error in Bodo Physical Builder's createJoin: Left and Right input do not have the same convention" }

        if (semiJoinDone) {
            throw UnsupportedOperationException("Semi-join operation is not supported")
        } else if (variablesSet.isNotEmpty()) {
            throw UnsupportedOperationException("Correlation variables are not supported")
        }

        val inputConvention = left.convention

        val retVal = if (inputConvention == PandasRel.CONVENTION) {
            PandasJoin.create(left, right, condition, joinType)
        } else if (inputConvention == SnowflakeRel.CONVENTION) {
            throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder's createJoin: unhandled Snowflake operation")
        } else if (inputConvention == IcebergRel.CONVENTION) {
            throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder's createJoin: unhandled Iceberg operation")
        } else {
            throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder: Unknown convention: " + inputConvention?.name)
        }

        assert(retVal.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE) == left.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)) { "TODO createProject: fix Batching property" }
        return retVal
    }

    private fun createSetOp(kind: SqlKind, inputs: List<RelNode>, all: Boolean): RelNode {
        assert(inputs.isNotEmpty()) { "Internal Error in Bodo Physical Builder: inputs to SetOp is empty" }
        val inputConvention = inputs[0].convention
        val cluster = inputs[0].cluster
        assert(inputConvention != null) { "Internal Error in Bodo Physical Builder: Input does not have any convention" }
        assert(inputs.all { input -> input.convention == inputConvention }) { "Internal Error in Bodo Physical Builder: inputs to SetOp have differing convention" }

        val retVal = if (inputConvention == PandasRel.CONVENTION) {
            createPandasSetOp(cluster, kind, inputs, all)
        } else if (inputConvention == SnowflakeRel.CONVENTION) {
            throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder's createPandasSetOp: unhandled snowflake set operation: " + kind.name)
        } else if (inputConvention == IcebergRel.CONVENTION) {
            throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder's createPandasSetOp: unhandled Iceberg set operation: " + kind.name)
        } else {
            throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder: Unknown convention: " + inputConvention?.name)
        }

        assert(retVal.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE) == inputs[0].traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)) { "TODO createProject: fix Batching property" }
        return retVal
    }

    private fun createPandasSetOp(cluster: RelOptCluster, kind: SqlKind, inputs: List<RelNode>, all: Boolean): RelNode {
        return when (kind) {
            SqlKind.UNION -> PandasUnion.create(cluster, inputs, all)
            // TODO: add the rest of the set operations
            else -> throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder's createPandasSetOp: unhandled pandas set operation: " + kind.name)
        }
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

        val strippedInput = if (input is HepRelVertex) {
            input.stripped()
        } else {
            input
        }

        val retVal = if (inputConvention == PandasRel.CONVENTION) {
            PandasAggregate.create(strippedInput.cluster, strippedInput, groupSet, groupSets, aggCalls)
        } else if (inputConvention == SnowflakeRel.CONVENTION) {
            SnowflakeAggregate.create(strippedInput.cluster, strippedInput.traitSet, strippedInput, groupSet, groupSets, aggCalls, (strippedInput as SnowflakeRel).getCatalogTable())
        } else if (inputConvention == IcebergRel.CONVENTION) {
            throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder's createAggregate: unhandled Iceberg operation")
        } else {
            throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder: Unknown convention: " + inputConvention?.name)
        }

        assert(retVal.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE) == strippedInput.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)) { "TODO createProject: fix Batching property" }
        return retVal
    }

    private fun createSort(input: RelNode, collation: RelCollation, offset: RexNode?, fetch: RexNode?): RelNode {
        val inputConvention = input.convention
        assert(inputConvention != null) { "Internal Error in Bodo Physical Builder: Input does not have any convention" }

        val strippedInput = if (input is HepRelVertex) {
            input.stripped()
        } else {
            input
        }

        val retVal = if (inputConvention == PandasRel.CONVENTION) {
            PandasSort.create(strippedInput, collation, offset, fetch)
        } else if (inputConvention == SnowflakeRel.CONVENTION) {
            SnowflakeSort.create(strippedInput.cluster, strippedInput.traitSet, strippedInput, collation, offset, fetch, (strippedInput as SnowflakeRel).getCatalogTable())
        } else if (inputConvention == IcebergRel.CONVENTION) {
            IcebergSort.create(strippedInput.cluster, strippedInput.traitSet, strippedInput, collation, offset, fetch, (strippedInput as IcebergRel).getCatalogTable())
        } else {
            throw BodoSQLCodegenException("Internal Error in Bodo Physical Builder: Unknown convention: " + inputConvention?.name)
        }

        assert(retVal.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE) == input.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)) { "TODO createProject: fix Batching property" }
        return retVal
    }

    private fun createTableScan(toRelContext: RelOptTable.ToRelContext, table: RelOptTable): RelNode {
        val bodoSqlTable: BodoSqlTable = ((table as? RelOptTableImpl)?.table() as? BodoSqlTable)!!

        val retVal = if (bodoSqlTable is SnowflakeCatalogTable) {
            SnowflakeTableScan.create(toRelContext.cluster, table, bodoSqlTable)
        } else {
            PandasTableScan.create(toRelContext.cluster, table)
        }
        return retVal
    }

    private fun createValues(
        cluster: RelOptCluster,
        rowType: RelDataType,
        tuples: List<ImmutableList<RexLiteral>>,
    ): RelNode {
        val immutableTuples = ImmutableList.copyOf(tuples)
        return PandasValues.create(cluster, cluster.traitSet().replace(PandasRel.CONVENTION), rowType, immutableTuples)
    }
}
