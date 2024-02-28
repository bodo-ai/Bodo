package com.bodosql.calcite.table

import com.bodosql.calcite.adapter.iceberg.IcebergTableScan
import com.bodosql.calcite.catalog.IcebergCatalog
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.rel.RelNode

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
}
