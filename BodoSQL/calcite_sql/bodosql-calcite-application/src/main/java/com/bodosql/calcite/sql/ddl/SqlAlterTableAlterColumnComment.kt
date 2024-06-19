package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * SqlNode for ALTER TABLE ALTER COLUMN COMMENT statement.
 * @param pos The parser position.
 * @param ifExists If the IF EXISTS clause is present. If true, will not error even if the table does not exist.
 * @param table The table which holds the column to modify.
 * @param column SqlIdentifier representing the column to modify.
 * @param comment: SqlNode representing the comment string. Should be `SqlLiteral`.
 */
class SqlAlterTableAlterColumnComment(
    pos: SqlParserPos,
    ifExists: Boolean,
    table: SqlIdentifier,
    column: SqlIdentifier,
    val comment: SqlNode,
) : SqlAlterTableAlterColumn(pos, ifExists, table, column) {
    override fun unparseAlterColumnStatement(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("COMMENT")
        comment.unparse(writer, leftPrec, rightPrec)
    }

    override fun getOperandList(): List<SqlNode> = listOf(table, column, comment)
}
