package com.bodosql.calcite.sql.ddl

import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.SqlCall
import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.SqlSpecialOperator
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.sql.type.ReturnTypes

/**
 * Parse tree node representing a {@code SHOW} clause.
 */
class SqlSnowflakeShowObjects(
    val pos: SqlParserPos,
    val schemaName: SqlIdentifier,
) : SqlCall(pos) {
    companion object {
        @JvmStatic
        private val OPERATOR: SqlOperator =
            SqlSpecialOperator(
                "SHOW OBJECTS",
                SqlKind.SHOW_OBJECTS,
                32,
                false,
                ReturnTypes.VARCHAR_2000,
                null,
                null,
            )
    }

    override fun getOperator(): SqlOperator = OPERATOR

    override fun getOperandList(): MutableList<SqlNode>? {
        return ImmutableList.of(schemaName)
    }

    override fun unparse(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("SHOW OBJECTS")
        schemaName.unparse(writer, leftPrec, rightPrec)
    }
}
