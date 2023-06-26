package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.ddl.SqlCreateTable
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * Abstract base class to describe any `CREATE TABLE` statement
 */
abstract class SqlSnowflakeCreateTableBase(
    pos: SqlParserPos?,
    replace: Boolean,
    tableType: CreateTableType?,
    ifNotExists: Boolean,
    name: SqlIdentifier?,
    columnList: SqlNodeList?,
    query: SqlNode?
) : SqlCreateTable(pos, replace, ifNotExists, name, columnList, query) {
    init {
        createType = tableType
    }

    override fun unparse(writer: SqlWriter, leftPrec: Int, rightPrec: Int) {
        writer.keyword("CREATE")
        if (replace) {
            writer.keyword("OR REPLACE")
        }
        if (!createTableType.equals(CreateTableType.DEFAULT)) {
            writer.keyword(createTableType.asStringKeyword());
        }
        writer.keyword("TABLE")
        if (ifNotExists) {
            writer.keyword("IF NOT EXISTS")
        }
        name.unparse(writer, leftPrec, rightPrec)
        unparseSuffix(writer, leftPrec, rightPrec)
    }

    abstract fun unparseSuffix(writer: SqlWriter, leftPrec: Int, rightPrec: Int);

}
