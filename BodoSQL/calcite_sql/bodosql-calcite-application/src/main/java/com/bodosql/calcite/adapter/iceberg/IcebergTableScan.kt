package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.table.CatalogTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.core.TableScan
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeField
import org.apache.calcite.rel.type.RelRecordType
import org.apache.calcite.util.ImmutableBitSet

class IcebergTableScan constructor(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    table: RelOptTable,
    val keptColumns: ImmutableBitSet,
    private val catalogTable: CatalogTable,
) :
    TableScan(cluster, traitSet.replace(IcebergRel.CONVENTION), ImmutableList.of(), table), IcebergRel {
        /**
         * This exists to set the type to the original names rather than
         * the lowercase normalized names that the table itself exposes.
         */
        override fun deriveRowType(): RelDataType {
            val fieldList = table.getRowType().fieldList
            val fields: MutableList<RelDataTypeField> = ArrayList()
            keptColumns.iterator().forEach { i -> fields.add(fieldList[i]) }
            return RelRecordType(fields)
        }

        // Update the digest to include listed columns
        override fun explainTerms(pw: RelWriter): RelWriter {
            val columnNames = deriveRowType().fieldNames
            return super.explainTerms(pw)
                .item("Columns", columnNames)
                .item("Iceberg", true)
        }

        override fun getCatalogTable(): CatalogTable = catalogTable

        override fun copy(
            traitSet: RelTraitSet,
            inputs: List<RelNode>,
        ): IcebergTableScan {
            return copy(traitSet, keptColumns)
        }

        private fun copy(
            traitSet: RelTraitSet,
            keptColumns: ImmutableBitSet,
        ): IcebergTableScan {
            return IcebergTableScan(cluster, traitSet, table, keptColumns, catalogTable)
        }

        /**
         * Create a new IcebergTableScan but pruning the existing columns. KeptColumns
         * will be given as the ImmutableBitSet of the current type, so we cannot simply
         * Intersect the two Bitsets.
         */
        fun cloneWithProject(newKeptColumns: ImmutableBitSet): IcebergTableScan {
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
         * Does this table scan include column pruning
         */
        fun prunesColumns(): Boolean {
            return keptColumns.cardinality() != table.getRowType().fieldCount
        }

        companion object {
            @JvmStatic
            fun create(
                cluster: RelOptCluster,
                table: RelOptTable,
                catalogTable: CatalogTable,
            ): IcebergTableScan {
                // Note: Types may be lazily computed so use getRowType() instead of rowType
                val rowType = table.getRowType()
                val keptColumns = ImmutableBitSet.range(rowType.fieldCount)
                return IcebergTableScan(cluster, cluster.traitSet(), table, keptColumns, catalogTable)
            }
        }
    }
