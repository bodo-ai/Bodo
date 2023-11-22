package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.table.SnowflakeCatalogTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.core.TableScan
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeField
import org.apache.calcite.rel.type.RelDataTypeFieldImpl
import org.apache.calcite.rel.type.RelRecordType
import org.apache.calcite.util.ImmutableBitSet

class SnowflakeTableScan private constructor(cluster: RelOptCluster, traitSet: RelTraitSet, table: RelOptTable, val keptColumns: ImmutableBitSet, private val catalogTable: SnowflakeCatalogTable) :
    TableScan(cluster, traitSet, ImmutableList.of(), table), SnowflakeRel {

    /**
     * This exists to set the type to the original names rather than
     * the lowercase normalized names that the table itself exposes.
     */
    override fun deriveRowType(): RelDataType {
        val fieldList = table.getRowType().fieldList
        val fields: MutableList<RelDataTypeField> = ArrayList()
        keptColumns.iterator().forEach {
                i ->
            val field = fieldList[i]
            val name = catalogTable.getPreservedColumnName(field.name)
            fields.add(RelDataTypeFieldImpl(name, field.index, field.type))
        }
        return RelRecordType(fields)
    }

    // Update the digest to include listed columns
    override fun explainTerms(pw: RelWriter): RelWriter {
        val columnNames = deriveRowType().fieldNames
        return super.explainTerms(pw)
            .item("Columns", columnNames)
    }

    override fun getCatalogTable(): SnowflakeCatalogTable = catalogTable

    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>): SnowflakeTableScan {
        return copy(traitSet, keptColumns)
    }

    private fun copy(traitSet: RelTraitSet, keptColumns: ImmutableBitSet): SnowflakeTableScan {
        return SnowflakeTableScan(cluster, traitSet, table, keptColumns, catalogTable)
    }

    /**
     * Create a new SnowflakeTableScan but pruning the existing columns. KeptColumns
     * will be given as the ImmutableBitSet of the current type, so we cannot simply
     * Intersect the two Bitsets.
     */
    fun cloneWithProject(newKeptColumns: ImmutableBitSet): SnowflakeTableScan {
        // Map column numbers to the original column numbers.
        // Convert to a list for fast lookup.
        val liveColumns = keptColumns.asList()
        val colsList: MutableList<Int> = ArrayList()
        for (colNumber in newKeptColumns.iterator()) {
            colsList.add(liveColumns[colNumber])
        }
        val finalColumns = ImmutableBitSet.of(colsList)
        return copy(traitSet, finalColumns)
    }

    /**
     * Get the original column index in the source table for the index
     * in the current type.
     */
    fun getOriginalColumnIndex(newIndex: Int): Int {
        // TODO(njriasan): Refactor to be more efficient
        // and/or cache the list.
        return keptColumns.asList()[newIndex]
    }

    /**
     * Does this table scan include column pruning
     */
    fun prunesColumns(): Boolean {
        return keptColumns.cardinality() != table.getRowType().fieldCount
    }

    override fun register(planner: RelOptPlanner) {
        planner.addRule(SnowflakeRules.TO_PANDAS)
        for (rule in SnowflakeRules.rules()) {
            planner.addRule(rule)
        }
    }

    companion object {
        @JvmStatic
        fun create(cluster: RelOptCluster, table: RelOptTable, catalogTable: SnowflakeCatalogTable): SnowflakeTableScan {
            // Note: Types may be lazily computed so use getRowType() instead of rowType
            val rowType = table.getRowType()
            val keptColumns = ImmutableBitSet.range(rowType.fieldCount)
            return SnowflakeTableScan(cluster, cluster.traitSet(), table, keptColumns, catalogTable)
        }
    }
}
