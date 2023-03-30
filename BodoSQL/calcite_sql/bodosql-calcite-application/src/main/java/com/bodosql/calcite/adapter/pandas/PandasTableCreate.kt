package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.LogicalTableCreate
import org.apache.calcite.schema.Schema

class PandasTableCreate(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    schema: Schema,
    tableName: String,
    isReplace: Boolean,
    path: List<String>,
) : LogicalTableCreate(cluster, traitSet, input, schema, tableName, isReplace, path), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>): PandasTableCreate {
        return PandasTableCreate(cluster, traitSet, sole(inputs),
            schema, tableName, isReplace, schemaPath)
    }

    override fun emit(builder: Module.Builder, inputs: () -> List<Dataframe>): Dataframe {
        TODO("Not yet implemented")
    }
}
