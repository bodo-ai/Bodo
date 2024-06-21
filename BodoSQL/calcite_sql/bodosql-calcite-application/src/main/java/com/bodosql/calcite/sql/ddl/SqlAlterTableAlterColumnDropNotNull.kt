package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * SqlNode for ALTER TABLE ALTER COLUMN DROP NOT NULL statement.
 * @param pos The parser position.
 * @param ifExists If the IF EXISTS clause is present. If true, will not error even if the table does not exist.
 * @param table The table which holds the column to modify.
 * @param column SqlIdentifier representing the column to change to nullable
 */
class SqlAlterTableAlterColumnDropNotNull(
    pos: SqlParserPos,
    ifExists: Boolean,
    table: SqlIdentifier,
    column: SqlIdentifier,
) : SqlAlterTableAlterColumn(pos, ifExists, table, column) {
    override fun unparseAlterColumnStatement(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("DROP")
        writer.keyword("NOT")
        writer.keyword("NULL")
    }

    override fun getOperandList(): List<SqlNode> = listOf(table, column)
}
