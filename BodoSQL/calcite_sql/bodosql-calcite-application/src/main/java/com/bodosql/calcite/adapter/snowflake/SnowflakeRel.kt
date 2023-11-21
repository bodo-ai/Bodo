package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.catalog.SnowflakeCatalogImpl
import com.bodosql.calcite.table.CatalogTable
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.rel2sql.BodoRelToSqlConverter
import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlSelect
import org.apache.calcite.sql.SqlWriterConfig
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.sql.util.SqlString
import java.util.function.UnaryOperator

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

    /**
     * If the node is simple enough to push down to snowflake, then this function will push the query to snowflake,
     * and return the output row count. Returns null if the query is too complex, or the query times out in SF.
     *
     * "simple enough" in this case is determined by shouldPushDownMetadataQueryHelper.shouldPushAsMetaDataQuery.
     *
     * @return the row count according to SF, or null
     */
    fun tryGetExpectedRowCountFromSFQuery(): Double? {
        if (!(shouldPushDownMetadataQueryHelper.shouldPushAsMetaDataQuery(this))) {
            return null
        }

        val rel2sql = BodoRelToSqlConverter(BodoSnowflakeSqlDialect.DEFAULT)
        val baseSqlNode = rel2sql.visitRoot(this).asFrom()

        // Add the count(*)
        val selectList = SqlNodeList.of(SqlStdOperatorTable.COUNT.createCall(SqlParserPos.ZERO, SqlIdentifier.STAR))

        val metadataSelectQuery = SqlSelect(
            SqlParserPos.ZERO,
            SqlNodeList.EMPTY,
            selectList,
            baseSqlNode,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
            null,
        )

        val metadataSelectQueryString: SqlString = metadataSelectQuery.toSqlString(
            UnaryOperator { c: SqlWriterConfig ->
                c.withClauseStartsLine(false)
                    .withDialect(BodoSnowflakeSqlDialect.NO_DOLLAR_ESCAPE)
            },
        )

        return this.getCatalogTable().trySubmitIntegerMetadataQuerySnowflake(metadataSelectQueryString)?.toDouble()
    }

    fun getCatalogTable(): CatalogTable

    /**
     * Get the batching property.
     */
    fun batchingProperty(): BatchingProperty = traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE) ?: BatchingProperty.NONE
}
