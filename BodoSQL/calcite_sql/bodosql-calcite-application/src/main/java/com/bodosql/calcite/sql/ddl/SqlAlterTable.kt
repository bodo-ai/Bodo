package com.bodosql.calcite.sql.ddl

import com.bodosql.calcite.sql.validate.BodoSqlValidator
import org.apache.calcite.sql.*
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.sql.validate.*


/*
 * Superclass for AlterTable statements. Currently supported features:
 * - ALTER TABLE table_name RENAME TO new_name
 * - ALTER TABLE table_name SWAP WITH other_table_name
 * - ALTER TABLE table_name ADD COLUMN col_name col_type
 * - ALTER TABLE table_name RENAME COLUMN col_name TO new_col_name
 * - ALTER TABLE table_name DROP COLUMN col_name1[, col_name2, col_name_3, ...]
 */
abstract class SqlAlterTable(
    val pos: SqlParserPos,
    val ifExists : Boolean,
    val table : SqlIdentifier,
) : SqlCall(pos) {

    companion object {
        @JvmField
        val OPERATOR = SqlSpecialOperator("ALTER TABLE", SqlKind.ALTER_TABLE)
    }
    override fun getOperator(): SqlOperator = OPERATOR

    override fun unparse(writer: SqlWriter, leftPrec: Int, rightPrec: Int)  {
            writer.keyword("ALTER")
            writer.keyword("TABLE")
            if (ifExists) {
                writer.keyword("IF")
                writer.keyword("EXISTS")
            }
            table.unparse(writer, leftPrec, rightPrec)
            unparseSuffix(writer, leftPrec, rightPrec)
    }

    abstract fun unparseSuffix(writer: SqlWriter, leftPrec: Int, rightPrec: Int)
}
