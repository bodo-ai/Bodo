package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.ir.BodoEngineTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.TableScan

class PandasTargetTableScan(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    table: RelOptTable,
) : TableScan(cluster, traitSet.replace(PandasRel.CONVENTION), ImmutableList.of(), table),
    PandasRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: MutableList<RelNode>?,
    ): RelNode = PandasTargetTableScan(cluster, traitSet, table)

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            relOptTable: RelOptTable,
        ): PandasTargetTableScan = PandasTargetTableScan(cluster, traitSet, relOptTable)

        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            relOptTable: RelOptTable,
        ): PandasTargetTableScan = create(cluster, cluster.traitSet(), relOptTable)
    }
}
