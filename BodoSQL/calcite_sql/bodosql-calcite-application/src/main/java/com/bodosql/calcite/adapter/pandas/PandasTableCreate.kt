package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.LogicalTableCreate
import org.apache.calcite.schema.Schema
import org.apache.calcite.sql.ddl.SqlCreateTable

class PandasTableCreate(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    schema: Schema,
    tableName: String,
    isReplace: Boolean,
    createTableType: SqlCreateTable.CreateTableType,
    path: List<String>,
) : LogicalTableCreate(cluster, traitSet, input, schema, tableName, isReplace, createTableType, path), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>): PandasTableCreate {
        return PandasTableCreate(cluster, traitSet, sole(inputs),
            schema, tableName, isReplace, createTableType, schemaPath)
    }

    override fun emit(builder: Module.Builder, inputs: () -> List<Dataframe>): Dataframe {
        TODO("Not yet implemented")
    }
}
