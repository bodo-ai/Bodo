package com.bodosql.calcite.adapter.snowflake

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

    fun generatePythonConnStr(schema: String): String
}
