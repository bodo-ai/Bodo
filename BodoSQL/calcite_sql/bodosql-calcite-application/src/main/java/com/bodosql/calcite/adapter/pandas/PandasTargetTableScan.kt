package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.table.BodoSqlTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.prepare.RelOptTableImpl
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.TableScan

class PandasTargetTableScan(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    table: RelOptTable,
) : TableScan(cluster, traitSet.replace(PandasRel.CONVENTION), ImmutableList.of(), table), PandasRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: MutableList<RelNode>?,
    ): RelNode {
        return PandasTargetTableScan(cluster, traitSet, table)
    }

    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    override fun getTimerType() = SingleBatchRelNodeTimer.OperationType.IO_BATCH

    override fun operationDescriptor() = "reading table"

    override fun loggingTitle() = "IO TIMING"

    override fun nodeDetails(): String {
        val relTable = table as RelOptTableImpl
        val bodoSqlTable = relTable.table() as BodoSqlTable
        return bodoSqlTable.name
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

    // Target table scans cannot use the node cache.
    override fun canUseNodeCache(): Boolean {
        return false
    }
}
