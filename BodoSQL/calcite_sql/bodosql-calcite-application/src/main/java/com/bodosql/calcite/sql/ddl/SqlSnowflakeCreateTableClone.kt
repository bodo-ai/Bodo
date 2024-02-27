package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * Object to describe a `CREATE TABLE` statement using `CLONE`
 */
class SqlSnowflakeCreateTableClone(
    pos: SqlParserPos?,
    replace: Boolean,
    tableType: CreateTableType?,
    ifNotExists: Boolean,
    name: SqlIdentifier?,
    val cloneSource: SqlNode,
    val copyGrants: Boolean,
    comment: SqlNode?,
) : SqlSnowflakeCreateTableBase(pos, replace, tableType, ifNotExists, name, null, cloneSource, comment) {
    override fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("CLONE")
        cloneSource.unparse(writer, 0, 0)
        if (copyGrants) writer.keyword("COPY GRANTS")
    }
}
