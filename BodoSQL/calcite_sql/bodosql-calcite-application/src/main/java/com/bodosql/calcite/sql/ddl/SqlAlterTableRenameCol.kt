package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

class SqlAlterTableRenameCol(
    pos: SqlParserPos,
    ifExists: Boolean,
    table: SqlIdentifier,
    val renameColOld: SqlIdentifier,
    val renameColNew: SqlIdentifier,
) : SqlAlterTable(pos, ifExists, table) {
    override fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("RENAME")
        writer.keyword("COLUMN")
        renameColOld.unparse(writer, leftPrec, rightPrec)
        writer.keyword("TO")
        renameColNew.unparse(writer, leftPrec, rightPrec)
    }

    override fun getOperandList(): List<SqlNode> = listOf(table, renameColOld, renameColNew)
}
