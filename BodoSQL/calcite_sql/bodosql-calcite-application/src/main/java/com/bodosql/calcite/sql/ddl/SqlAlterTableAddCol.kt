package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * SqlNode for ALTER TABLE ADD COLUMN statement.
 * @param pos The parser position.
 * @param ifExists If the IF EXISTS clause is present. If true, will not error even if the table does not exist.
 * @param ifNotExists If the IF NOT EXISTS clause for the column is present. If true, will not error if trying to
 *                    create already existing column
 * @param table The table to add a column to.
 * @param addCol The SqlNode containing information about the new column. Should be of type SqlSnowflakeColumnDeclaration.
 */
class SqlAlterTableAddCol(
    pos: SqlParserPos,
    ifExists: Boolean,
    val ifNotExists: Boolean,
    table: SqlIdentifier,
    val addCol: SqlNode,
) : SqlAlterTable(pos, ifExists, table) {
    override fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("ADD")
        writer.keyword("COLUMN")
        if (ifNotExists) {
            writer.keyword("IF")
            writer.keyword("NOT")
            writer.keyword("EXISTS")
        }
        addCol.unparse(writer, leftPrec, rightPrec)
    }

    override fun getOperandList(): List<SqlNode> = listOf(table, addCol)
}
