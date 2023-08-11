package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.table.CatalogTableImpl
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
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

class SnowflakeTableScan private constructor(cluster: RelOptCluster, traitSet: RelTraitSet, table: RelOptTable?, private val catalogTable: CatalogTableImpl) :
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

    override fun getCatalogTable(): CatalogTableImpl = catalogTable

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>): RelNode {
        return SnowflakeTableScan(cluster, traitSet, table, catalogTable)
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
            // Note: Types may be lazily computed so use getRowType() instead of rowType
            val batchingProperty = ExpectedBatchingProperty.streamingIfPossibleProperty(table.getRowType())
            val traitSet = cluster.traitSet().replace(batchingProperty)
            return SnowflakeTableScan(cluster, traitSet, table, catalogTable)
        }
    }
}
