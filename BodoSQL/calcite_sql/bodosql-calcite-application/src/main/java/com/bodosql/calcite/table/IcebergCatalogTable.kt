package com.bodosql.calcite.table

import com.bodosql.calcite.adapter.iceberg.IcebergTableScan
import com.bodosql.calcite.catalog.IcebergCatalog
import com.google.common.base.Supplier
import com.google.common.base.Suppliers
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.rel.RelNode
import org.apache.calcite.schema.Statistic

class IcebergCatalogTable(
    name: String,
    schemaPath: ImmutableList<String>,
    columns: List<BodoSQLColumn>,
    private val catalog: IcebergCatalog,
) : CatalogTable(
        name,
        schemaPath,
        columns,
        catalog,
    ) {
    // Hold the statistics for this table.
    private val statistic: Statistic = StatisticImpl()

    /** Interface to get the Iceberg Catalog.  */
    override fun getCatalog(): IcebergCatalog {
        return catalog
    }

    override fun toRel(
        toRelContext: RelOptTable.ToRelContext,
        relOptTable: RelOptTable,
    ): RelNode? {
        return IcebergTableScan.create(toRelContext.cluster, relOptTable, this)
    }

    override fun getStatistic(): Statistic {
        return statistic
    }

    private inner class StatisticImpl : Statistic {
        private val rowCount: Supplier<Double?> = Suppliers.memoize { estimateRowCount() }

        /**
         * Retrieves the estimated row count for this table. This value is memoized.
         *
         * @return estimated row count for this table.
         */
        override fun getRowCount(): Double? {
            return rowCount.get()
        }

        /**
         * Retrieves the estimated row count for this table. It performs a query every time this is
         * invoked.
         *
         * @return estimated row count for this table.
         */
        private fun estimateRowCount(): Double? {
            return catalog.estimateIcebergTableRowCount(parentFullPath, name)
        }
    }
}
