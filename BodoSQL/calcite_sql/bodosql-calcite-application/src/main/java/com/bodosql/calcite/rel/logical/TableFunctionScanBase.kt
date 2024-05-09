package com.bodosql.calcite.rel.core

import com.bodosql.calcite.adapter.bodo.BodoPhysicalTableFunctionScan
import com.bodosql.calcite.application.operatorTables.TableFunctionOperatorTable
import com.bodosql.calcite.application.operatorTables.TableFunctionOperatorTable.EXTERNAL_TABLE_FILES_NAME
import com.bodosql.calcite.plan.makeCost
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.TableFunctionScan
import org.apache.calcite.rel.metadata.RelColumnMapping
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import java.lang.reflect.Type
import java.math.BigDecimal

/**
 * Base table function scan node for defining the cost model.
 */
open class TableFunctionScanBase(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    inputs: List<RelNode>,
    call: RexCall,
    elementType: Type?,
    rowType: RelDataType,
    columnMappings: Set<RelColumnMapping>?,
) : TableFunctionScan(cluster, traits, inputs, call, elementType, rowType, columnMappings) {
    override fun copy(
        traitSet: RelTraitSet?,
        inputs: MutableList<RelNode>?,
        rexCall: RexNode?,
        elementType: Type?,
        rowType: RelDataType?,
        columnMappings: MutableSet<RelColumnMapping>?,
    ): TableFunctionScan? {
        return BodoPhysicalTableFunctionScan(cluster, traitSet!!, inputs!!, rexCall!! as RexCall, elementType, rowType!!, columnMappings)
    }

    override fun estimateRowCount(mq: RelMetadataQuery): Double {
        val rexCall = call as RexCall
        if (rexCall.op.name == TableFunctionOperatorTable.GENERATOR.name) {
            // For GENERATOR, the first argument is the row count
            val rowCount = rexCall.operands[0] as RexLiteral
            return rowCount.getValueAs(BigDecimal::class.java)!!.toDouble()
        } else if (rexCall.op.name == EXTERNAL_TABLE_FILES_NAME) {
            // For EXTERNAL_TABLE_FILES, use a small number:
            return 16.0
        }
        return super.estimateRowCount(mq)
    }

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        val rows = estimateRowCount(mq)
        return planner.makeCost(rows = rows)
    }
}
