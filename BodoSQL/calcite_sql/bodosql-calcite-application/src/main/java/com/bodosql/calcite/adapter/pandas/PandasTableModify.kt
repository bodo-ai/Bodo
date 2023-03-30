package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.prepare.Prepare
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

    override fun emit(builder: Module.Builder, inputs: () -> List<Dataframe>): Dataframe {
        TODO("Not yet implemented")
    }
}
