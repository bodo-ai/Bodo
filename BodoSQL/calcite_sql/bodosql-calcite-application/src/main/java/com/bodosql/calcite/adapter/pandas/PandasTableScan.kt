package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.Dataframe
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.core.TableScan

class PandasTableScan(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    table: RelOptTable,
) : TableScan(cluster, traitSet, ImmutableList.of(), table), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun emit(implementor: PandasRel.Implementor): Dataframe {
        TODO("Not yet implemented")
    }
}
