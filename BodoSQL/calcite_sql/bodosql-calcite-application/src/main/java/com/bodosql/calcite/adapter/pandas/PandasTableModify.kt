package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.table.BodoSqlTable
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
) : TableModify(cluster, traitSet, table, catalogReader,
    input, operation, updateColumnList, sourceExpressionList, flattened), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>): PandasTableModify {
        return PandasTableModify(cluster, traitSet, table, catalogReader,
            sole(inputs), operation, updateColumnList, sourceExpressionList, isFlattened)
    }

    override fun emit(implementor: PandasRel.Implementor): Dataframe {
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

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
        TODO("Not yet implemented")
    }
}
