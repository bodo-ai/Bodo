package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

class SqlAlterTableDropCol(
    pos: SqlParserPos,
    ifExists: Boolean,
    table: SqlIdentifier,
    val dropCols: SqlNodeList,
) : SqlAlterTable(pos, ifExists, table) {
    override fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("DROP")
        writer.keyword("COLUMN")
        dropCols.unparse(writer, leftPrec, rightPrec)
    }

    override fun getOperandList(): List<SqlNode> = listOf(table, dropCols)
}
