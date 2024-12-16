package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * SqlNode for ALTER TABLE RENAME TO statement.
 * ifExists parameter is responsible for optional IF EXISTS clause.
 * Subclass of SqlAlterTable.
 */

class SqlAlterTableRenameTable(
    pos: SqlParserPos,
    ifExists: Boolean,
    table: SqlIdentifier,
    val renameName: SqlIdentifier,
) : SqlAlterTable(pos, ifExists, table) {
    override fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("RENAME")
        writer.keyword("TO")
        renameName.unparse(writer, leftPrec, rightPrec)
    }

    override fun getOperandList(): List<SqlNode> = listOf(table, renameName)
}
