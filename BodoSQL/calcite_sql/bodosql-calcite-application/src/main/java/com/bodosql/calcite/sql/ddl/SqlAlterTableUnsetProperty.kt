package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * SqlNode for ALTER TABLE UNSET PROPERTY statement.
 * @param pos The parser position.
 * @param ifExists If the IF EXISTS clause for the table is present.
 * @param table The table to unset properties on.
 * @param propertyList The list of properties to unset. Must be a list of SqlLiterals.
 * @param ifPropertyExists If the IF EXISTS clause for the properties are present. If set to true,
 *                         the operation will not fail even if the property does not exist on the table.
 */

class SqlAlterTableUnsetProperty(
    pos: SqlParserPos,
    ifExists: Boolean,
    table: SqlIdentifier,
    val propertyList: SqlNodeList,
    val ifPropertyExists: Boolean,
) : SqlAlterTable(pos, ifExists, table) {
    override fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        val setFrame = writer.startList(SqlWriter.FrameTypeEnum.UPDATE_SET_LIST, "UNSET", "")
        writer.keyword("PROPERTY") // All keywords will just parse to PROPERTY
        for (_property in propertyList) {
            writer.sep(",")
            val property = _property as SqlLiteral
            property.unparse(writer, leftPrec, rightPrec)
        }
        writer.endList(setFrame)
    }

    override fun getOperandList(): List<SqlNode> = listOf(table, propertyList)
}
