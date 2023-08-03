package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.catalog.SnowflakeCatalogImpl
import com.bodosql.calcite.table.CatalogTableImpl
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
interface SnowflakeRel : RelNode {
    companion object {
        @JvmField
        val CONVENTION = Convention.Impl("SNOWFLAKE", SnowflakeRel::class.java)
    }

    fun generatePythonConnStr(schema: String): String {
        // TODO(jsternberg): The catalog will specifically be SnowflakeCatalogImpl.
        // This cast is a bad idea and is particularly unsafe and unverifiable using
        // the compiler tools. It would be better if the catalog implementations were
        // refactored to not be through an interface and we had an actual class type
        // that referenced snowflake than needing to do it through a cast.
        // That's a bit too much work to refactor quite yet, so this cast gets us
        // through this time where the code is too abstract and we just need a way
        // to convert over.
        val catalog = getCatalogTable().catalog as SnowflakeCatalogImpl
        return catalog.generatePythonConnStr(schema)
    }

    fun getCatalogTable(): CatalogTableImpl

}
