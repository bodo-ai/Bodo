package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlAlter
import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.SqlSpecialOperator
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/*
 * Superclass SqlNode for AlterTable statements.
 * Inherits from SqlAlter (which for some reason is not a DDLNode...)
 * To add support for more ALTER TABLE operations, subclass this class.
 *
 * Currently supported features:
 * - ALTER TABLE table_name RENAME TO new_name
 * - ALTER TABLE table_name SWAP WITH other_table_name
 * - ALTER TABLE table_name ADD COLUMN col_name col_type
 * - ALTER TABLE table_name RENAME COLUMN col_name TO new_col_name
 * - ALTER TABLE table_name DROP COLUMN col_name1[, col_name2, col_name_3, ...]
 */
abstract class SqlAlterTable(
    val pos: SqlParserPos,
    val ifExists: Boolean,
    val table: SqlIdentifier,
) : SqlAlter(pos) {
    companion object {
        @JvmField
        val OPERATOR = SqlSpecialOperator("ALTER TABLE", SqlKind.ALTER_TABLE)
    }

    override fun getOperator(): SqlOperator = OPERATOR

    override fun unparseAlterOperation(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("ALTER")
        writer.keyword("TABLE")
        if (ifExists) {
            writer.keyword("IF")
            writer.keyword("EXISTS")
        }
        table.unparse(writer, leftPrec, rightPrec)
        unparseSuffix(writer, leftPrec, rightPrec)
    }

    abstract fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    )
}
