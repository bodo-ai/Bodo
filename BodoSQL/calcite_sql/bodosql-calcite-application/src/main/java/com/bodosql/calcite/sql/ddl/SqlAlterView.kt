package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlAlter
import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.SqlSpecialOperator
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/*
 * Superclass SqlNode for AlterView statements.
 * Inherits from SqlAlter (which for some reason is not a DDLNode...)
 * To add support for more ALTER VIEW operations, subclass this class.
 *
 * Currently only supports RENAME TO, as follows:
 * - ALTER VIEW view_name RENAME TO new_name
 */
abstract class SqlAlterView(
    val pos: SqlParserPos,
    val ifExists: Boolean,
    val view: SqlIdentifier,
) : SqlAlter(pos) {
    companion object {
        @JvmField
        val OPERATOR = SqlSpecialOperator("ALTER VIEW", SqlKind.ALTER_VIEW)
    }

    override fun getOperator(): SqlOperator = OPERATOR

    override fun unparseAlterOperation(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("ALTER")
        writer.keyword("VIEW")
        if (ifExists) {
            writer.keyword("IF")
            writer.keyword("EXISTS")
        }
        view.unparse(writer, leftPrec, rightPrec)
        unparseSuffix(writer, leftPrec, rightPrec)
    }

    abstract fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    )
}
