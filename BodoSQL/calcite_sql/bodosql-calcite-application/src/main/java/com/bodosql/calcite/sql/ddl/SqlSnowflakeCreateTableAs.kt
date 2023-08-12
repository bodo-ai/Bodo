package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * Object to describe a `CREATE TABLE` statement using `AS SELECT`
 */
class SqlSnowflakeCreateTableAs(
    pos: SqlParserPos?,
    replace: Boolean,
    tableType: CreateTableType?,
    ifNotExists: Boolean,
    name: SqlIdentifier?,
    columnList: SqlNodeList?,
    val selectSource: SqlNode,
    val clusterExprs: SqlNodeList?,
    val copyGrants: Boolean,
) : SqlSnowflakeCreateTableBase(pos, replace, tableType, ifNotExists, name, columnList, selectSource) {
    init {
        // CREATE TABLE AS SELECT only allows COPY GRANTS if OR REPLACE is also provided
        if (copyGrants && !replace) {
            throw Exception("CREATE TABLE with AS SELECT requires OR REPLACE to use COPY GRANTS")
        }
    }

    override fun unparseSuffix(writer: SqlWriter, leftPrec: Int, rightPrec: Int) {
        getcolumnList()?.let {
            val frame = writer.startList("(", ")")
            for (c in it) {
                writer.sep(",")
                c.unparse(writer, 0, 0)
            }
            writer.endList(frame)
        }
        clusterExprs?.let {
            writer.keyword("CLUSTER BY")
            val frame = writer.startList("(", ")")
            clusterExprs.unparse(writer, leftPrec, rightPrec)
            writer.endList(frame)
        }
        if (copyGrants) writer.keyword("COPY GRANTS")
        writer.keyword("AS")
        writer.newlineAndIndent()
        selectSource.unparse(writer, 0, 0)
    }
}
