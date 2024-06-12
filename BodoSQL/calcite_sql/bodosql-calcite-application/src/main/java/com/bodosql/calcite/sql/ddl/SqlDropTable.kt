package com.bodosql.calcite.sql.ddl

import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.SqlDrop
import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.SqlSpecialOperator
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * Represents the DROP TABLE DDL command in SQL.
 *
 * This is an expansion of the one within Calcite using the same SqlKind
 * as this one also supports specifying CASCADE or RESTRICT.
 */
class SqlDropTable(
    pos: SqlParserPos,
    ifExists: Boolean,
    val name: SqlIdentifier,
    val cascade: Boolean,
    val purge: Boolean,
) : SqlDrop(OPERATOR, pos, ifExists) {
    companion object {
        @JvmStatic
        private val OPERATOR: SqlOperator =
            SqlSpecialOperator("DROP TABLE", SqlKind.DROP_TABLE)
    }

    override fun getOperandList(): List<SqlNode> = ImmutableList.of(name)

    override fun unparse(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword(operator.name)
        if (ifExists) {
            writer.keyword("IF EXISTS")
        }
        name.unparse(writer, leftPrec, rightPrec)
        writer.keyword(
            if (cascade) "CASCADE" else "RESTRICT",
        )
        if (purge) {
            writer.keyword("PURGE")
        }
    }
}
