package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.catalog.SnowflakeCatalogImpl
import com.bodosql.calcite.table.CatalogTableImpl
import com.bodosql.calcite.traits.BatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.TableScan
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFieldImpl
import org.apache.calcite.rel.type.RelRecordType

class SnowflakeTableScan private constructor(cluster: RelOptCluster, traitSet: RelTraitSet, table: RelOptTable?, val catalogTable: CatalogTableImpl) :
    TableScan(cluster, traitSet, ImmutableList.of(), table), SnowflakeRel {

    /**
     * This exists to set the type to the original names rather than
     * the lowercase normalized names that the table itself exposes.
     */
    override fun deriveRowType(): RelDataType {
        return RelRecordType(table.rowType.fieldList.map { field ->
            val name = catalogTable.getPreservedColumnName(field.name)
            RelDataTypeFieldImpl(name, field.index, field.type)
        })
    }

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>): RelNode {
        return SnowflakeTableScan(cluster, traitSet, table, catalogTable)
    }

    override fun generatePythonConnStr(schema: String): String {
        // TODO(jsternberg): The catalog will specifically be SnowflakeCatalogImpl.
        // This cast is a bad idea and is particularly unsafe and unverifiable using
        // the compiler tools. It would be better if the catalog implementations were
        // refactored to not be through an interface and we had an actual class type
        // that referenced snowflake than needing to do it through a cast.
        // That's a bit too much work to refactor quite yet, so this cast gets us
        // through this time where the code is too abstract and we just need a way
        // to convert over.
        val catalog = catalogTable.catalog as SnowflakeCatalogImpl
        return catalog.generatePythonConnStr(schema)
    }

    override fun register(planner: RelOptPlanner) {
        planner.addRule(SnowflakeRules.TO_PANDAS)
        for (rule in SnowflakeRules.rules()) {
            planner.addRule(rule)
        }
    }

    companion object {
        @JvmStatic
        fun create(cluster: RelOptCluster, table: RelOptTable, catalogTable: CatalogTableImpl): SnowflakeTableScan {
            // TODO(jsternberg): This next line should be required and always part
            // of creating a SnowflakeTableScan. On the other hand, while we are using the
            // HepPlanner, this trait set causes an issue. The existence of the trait set
            // causes the HepPlanner to try and stick to the convention, but it uses the
            // VolcanoPlanner initialized with PlannerImpl to do that. The VolcanoPlanner
            // gets confused about HepRelVertex and this causes it to fail to enforce
            // the traits.
            //
            // Until we're completely using the VolcanoPlanner for everything in general,
            // we'll avoid adding the trait set when creating the table scan so this is
            // treated like the NONE convention by the HepPlanner. We'll then add it using
            // copy and a RelShuttle before invoking the VolcanoPlanner.
            // val traitSet = cluster.traitSetOf(SnowflakeRel.CONVENTION)
            val traitSet = cluster.traitSet().replace(BatchingProperty.SINGLE_BATCH)
            return SnowflakeTableScan(cluster, traitSet, table, catalogTable)
        }
    }
}
