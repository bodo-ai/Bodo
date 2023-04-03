package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.catalog.SnowflakeCatalogImpl
import com.bodosql.calcite.table.CatalogTableImpl
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rex.RexNode

class SnowflakeFilter private constructor(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    condition: RexNode,
    val catalogTable: CatalogTableImpl,
) : Filter(cluster, traitSet, input, condition), SnowflakeRel {

    override fun copy(traitSet: RelTraitSet, input: RelNode, condition: RexNode): Filter {
        return SnowflakeFilter(cluster, traitSet, input, condition, catalogTable)
    }

    override fun generatePythonConnStr(schema: String): String {
        val catalog = catalogTable.catalog as SnowflakeCatalogImpl
        return catalog.generatePythonConnStr(schema)
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster, traitSet: RelTraitSet, input: RelNode,
            condition: RexNode, catalogTable: CatalogTableImpl
        ): SnowflakeFilter {
            val newTraitSet = traitSet.replace(SnowflakeRel.CONVENTION)
            return SnowflakeFilter(cluster, newTraitSet, input, condition, catalogTable)
        }
    }
}
