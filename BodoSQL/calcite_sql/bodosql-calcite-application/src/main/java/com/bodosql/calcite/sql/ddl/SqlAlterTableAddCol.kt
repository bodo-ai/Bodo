package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

class SqlAlterTableAddCol(
    pos: SqlParserPos,
    ifExists: Boolean,
    table: SqlIdentifier,
    val addCol: SqlNode,
) : SqlAlterTable(pos, ifExists, table) {
    override fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("ADD")
        writer.keyword("COLUMN")
        addCol.unparse(writer, leftPrec, rightPrec)
    }

    override fun getOperandList(): List<SqlNode> = listOf(table, addCol)
}
