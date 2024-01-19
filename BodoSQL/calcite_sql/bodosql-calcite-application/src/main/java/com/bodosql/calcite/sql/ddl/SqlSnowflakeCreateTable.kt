package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * Object to describe a regular `CREATE TABLE` statement
 */
class SqlSnowflakeCreateTable(
    pos: SqlParserPos?,
    replace: Boolean,
    tableType: CreateTableType?,
    ifNotExists: Boolean,
    name: SqlIdentifier?,
    columnList: SqlNodeList?,
    val clusterExprs: SqlNodeList?,
    val copyGrants: Boolean,
    comment: SqlNode?,
) : SqlSnowflakeCreateTableBase(pos, replace, tableType, ifNotExists, name, columnList, null, comment) {
    val columnComments: List<SqlNode?>?
    init {
        // Regular CREATE TABLE only allows COPY GRANTS if OR REPLACE is also provided
        if (copyGrants && !replace) {
            throw Exception("Regular CREATE TABLE requires OR REPLACE to use COPY GRANTS")
        }
        columnComments = columnList?.list?.map { (it as SqlSnowflakeColumnDeclaration).comment }
    }

    override fun unparseSuffix(writer: SqlWriter, leftPrec: Int, rightPrec: Int) {
        clusterExprs?.let {
            writer.keyword("CLUSTER BY")
            val frame = writer.startList("(", ")")
            clusterExprs.unparse(writer, leftPrec, rightPrec)
            writer.endList(frame)
        }
        if (copyGrants) writer.keyword("COPY GRANTS")
    }
}
