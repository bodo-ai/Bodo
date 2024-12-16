package com.bodosql.calcite.sql.ddl

import com.google.common.collect.ImmutableList
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
    pos: SqlParserPos,
    val schemaName: SqlIdentifier,
    isTerse: Boolean,
) : SqlShow(pos, isTerse) {
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

    override fun getOperandList(): MutableList<SqlNode>? = ImmutableList.of(schemaName)

    /**
     * This begins unparsing the clause after SHOW as the superclass
     * already deals with the SHOW statement.
     */
    override fun unparseShowOperation(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        // May need to raise the IN keyword into a boolean
        // when later adding support for SHOW without specifying schema/db
        writer.keyword("OBJECTS")
        writer.keyword("IN")
        schemaName.unparse(writer, leftPrec, rightPrec)
    }
}
