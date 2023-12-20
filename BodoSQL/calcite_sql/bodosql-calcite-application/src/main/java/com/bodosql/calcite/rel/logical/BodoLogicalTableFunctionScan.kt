package com.bodosql.calcite.rel.logical

import com.bodosql.calcite.rel.core.TableFunctionScanBase
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.TableFunctionScan
import org.apache.calcite.rel.metadata.RelColumnMapping
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexNode
import java.lang.reflect.Type

class BodoLogicalTableFunctionScan(cluster: RelOptCluster, traits: RelTraitSet, inputs: List<RelNode>, call: RexCall, elementType: Type?, rowType: RelDataType, columnMappings: Set<RelColumnMapping>?) : TableFunctionScanBase(cluster, traits, inputs, call, elementType, rowType, columnMappings) {
    override fun copy(
        traitSet: RelTraitSet?,
        inputs: MutableList<RelNode>?,
        rexCall: RexNode?,
        elementType: Type?,
        rowType: RelDataType?,
        columnMappings: MutableSet<RelColumnMapping>?,
    ): TableFunctionScan? {
        return BodoLogicalTableFunctionScan(cluster, traitSet!!, inputs!!, rexCall!! as RexCall, elementType, rowType!!, columnMappings)
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            inputs: List<RelNode>,
            call: RexCall,
            rowType: RelDataType,
        ): BodoLogicalTableFunctionScan {
            return BodoLogicalTableFunctionScan(cluster, cluster.traitSet(), inputs, call, null, rowType, null)
        }
    }
}
