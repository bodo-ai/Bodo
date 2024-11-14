package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.SqlSpecialOperator
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.util.ImmutableNullableList

/**
 * SqlNode for SHOW TBLPROPERTIES <tbl_name> ('property') SQL statement. Subclass of SqlShow.
 * @param pos The parser position.
 * @param table The table to show the properties of.
 * @param property A SqlLiteral representing the specific property to show. If this is null, the statement is
 *                 simply `SHOW TBLPROPERTIES` without specifying a property, which means show all properties.
 */
class SqlShowTblproperties(
    pos: SqlParserPos,
    val table: SqlIdentifier,
    // propertyName null if not passed in.
    val property: SqlLiteral?,
) : SqlShow(pos, false) {
    companion object {
        @JvmStatic
        private val OPERATOR: SqlOperator =
            SqlSpecialOperator(
                "SHOW TBLPROPERTIES",
                SqlKind.SHOW_TBLPROPERTIES,
            )
    }

    override fun getOperator(): SqlOperator = OPERATOR

    override fun getOperandList(): MutableList<SqlNode>? = ImmutableNullableList.of(table, property)

    /**
     * This begins unparsing the clause after SHOW as the superclass
     * already deals with the SHOW statement.
     */
    override fun unparseShowOperation(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("TBLPROPERTIES")
        table.unparse(writer, leftPrec, rightPrec)
        if (property != null) {
            writer.keyword("(")
            property.unparse(writer, leftPrec, rightPrec)
            writer.keyword(")")
        }
    }
}
