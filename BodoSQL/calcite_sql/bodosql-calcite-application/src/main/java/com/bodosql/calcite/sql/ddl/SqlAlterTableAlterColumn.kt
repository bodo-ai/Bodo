package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * Superclass SqlNode for `ALTER TABLE ALTER COLUMN` statements.
 * Inherits from SqlAlterTable
 * To add support for more `ALTER TABLE ALTER COLUMN` operations, subclass this class.
 *
 * Currently supported features:
 * `ALTER TABLE <table_name> ALTER COLUMN <col_name>...`
 * - `COMMENT 'comment_string'`
 *
 *  @param pos The parser position.
 *  @param ifExists If the IF EXISTS clause is present. If true, will not error even if the table does not exist.
 *  @param table The table which holds the column to modify.
 *  @param column SqlIdentifier representing the column to modify.
 */
abstract class SqlAlterTableAlterColumn(
    pos: SqlParserPos,
    ifExists: Boolean,
    table: SqlIdentifier,
    val column: SqlIdentifier,
) : SqlAlterTable(pos, ifExists, table) {
    /**
     * Parses up to `ALTER COLUMN <col_name>`.
     */
    override fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("ALTER")
        writer.keyword("COLUMN")
        column.unparse(writer, leftPrec, rightPrec)
        unparseAlterColumnStatement(writer, leftPrec, rightPrec)
    }

    /**
     * Parses the rest of the ALTER COLUMN statement.
     */
    abstract fun unparseAlterColumnStatement(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    )
}
