package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.util.Pair

/**
 * SqlNode for ALTER TABLE SET PROPERTY statement.
 * @param pos The parser position.
 * @param ifExists If the IF EXISTS clause is present.
 * @param table The table to set properties on.
 * @param propertyList The list of properties to set. Must be a list of SqlLiterals.
 * @param valueList The list of values to set. Must be a list of SqlLiterals.
 */

class SqlAlterTableSetProperty(
    pos: SqlParserPos,
    ifExists: Boolean,
    table: SqlIdentifier,
    val propertyList: SqlNodeList,
    val valueList: SqlNodeList,
) : SqlAlterTable(pos, ifExists, table) {
    override fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        val setFrame = writer.startList(SqlWriter.FrameTypeEnum.UPDATE_SET_LIST, "SET", "")
        writer.keyword("PROPERTY") // All keywords will just parse to PROPERTY
        for (pair in Pair.zip(propertyList, valueList)) {
            writer.sep(",")
            val id = pair.left as SqlLiteral
            id.unparse(writer, leftPrec, rightPrec)
            writer.keyword("=")
            val sourceExp = pair.right
            sourceExp.unparse(writer, leftPrec, rightPrec)
        }
        writer.endList(setFrame)
    }

    override fun getOperandList(): List<SqlNode> = listOf(table, propertyList, valueList)
}
