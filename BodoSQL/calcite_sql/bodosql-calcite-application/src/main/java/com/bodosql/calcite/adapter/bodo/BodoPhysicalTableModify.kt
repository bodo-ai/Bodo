package com.bodosql.calcite.adapter.bodo

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

class BodoPhysicalTableModify(
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
        cluster,
        traitSet.replace(BodoPhysicalRel.CONVENTION),
        table,
        catalogReader,
        input,
        operation,
        updateColumnList,
        sourceExpressionList,
        flattened,
    ),
    BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): BodoPhysicalTableModify =
        BodoPhysicalTableModify(
            cluster,
            traitSet,
            table,
            catalogReader,
            sole(inputs),
            operation,
            updateColumnList,
            sourceExpressionList,
            isFlattened,
        )

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
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

    override fun isStreaming() = (input as BodoPhysicalRel).batchingProperty() == BatchingProperty.STREAMING

    override fun expectedInputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        val bodoSqlTable = (table as RelOptTableImpl).table() as BodoSqlTable
        // Note: Types may be lazily computed so use getRowType() instead of rowType
        return ExpectedBatchingProperty.tableModifyProperty(bodoSqlTable, input.getRowType())
    }

    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        TODO("Not yet implemented")
    }
}
