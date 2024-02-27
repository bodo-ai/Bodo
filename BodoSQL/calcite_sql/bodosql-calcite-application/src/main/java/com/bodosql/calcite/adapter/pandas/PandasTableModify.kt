package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.table.BodoSqlTable
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.prepare.Prepare
import org.apache.calcite.prepare.RelOptTableImpl
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.TableModify
import org.apache.calcite.rex.RexNode

class PandasTableModify(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    table: RelOptTable,
    catalogReader: Prepare.CatalogReader,
    input: RelNode,
    operation: Operation,
    updateColumnList: List<String>?,
    sourceExpressionList: List<RexNode>?,
    flattened: Boolean,
) : TableModify(
        cluster, traitSet.replace(PandasRel.CONVENTION), table, catalogReader,
        input, operation, updateColumnList, sourceExpressionList, flattened,
    ),
    PandasRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): PandasTableModify {
        return PandasTableModify(
            cluster, traitSet, table, catalogReader,
            sole(inputs), operation, updateColumnList, sourceExpressionList, isFlattened,
        )
    }

    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    override fun getTimerType() = SingleBatchRelNodeTimer.OperationType.IO_BATCH

    override fun operationDescriptor() = "writing table"

    override fun loggingTitle() = "IO TIMING"

    override fun nodeDetails(): String {
        val relTable = table as RelOptTableImpl
        val bodoSqlTable = relTable.table() as BodoSqlTable
        return bodoSqlTable.name
    }

    override fun isStreaming() = (input as PandasRel).batchingProperty() == BatchingProperty.STREAMING

    override fun expectedInputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        val bodoSqlTable = (table as RelOptTableImpl).table() as BodoSqlTable
        // Note: Types may be lazily computed so use getRowType() instead of rowType
        return ExpectedBatchingProperty.tableModifyProperty(bodoSqlTable, input.getRowType())
    }

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(
        ctx: PandasRel.BuildContext,
        stateVar: StateVariable,
    ) {
        TODO("Not yet implemented")
    }
}
