package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.table.CatalogTable
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode

/**
 * Temporary convention for Snowflake relations.
 *
 * We should probably try to see if we can re-use JdbcRel
 * as that will also potentially open access to other databases
 * that we can use from Calcite, but there's really no need for it
 * right now as it doesn't help us with the existing code.
 */
interface IcebergRel : RelNode {
    companion object {
        @JvmField
        val CONVENTION = Convention.Impl("ICEBERG", IcebergRel::class.java)
    }

    fun generatePythonConnStr(schemaPath: ImmutableList<String>): Expr {
        return getCatalogTable().generatePythonConnStr(schemaPath)
    }

    fun getCatalogTable(): CatalogTable

    /**
     * Get the batching property.
     */
    fun batchingProperty(): BatchingProperty = traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE) ?: BatchingProperty.NONE
}
