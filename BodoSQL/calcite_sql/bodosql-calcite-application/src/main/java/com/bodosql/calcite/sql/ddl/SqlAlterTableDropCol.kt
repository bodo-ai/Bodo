package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * SqlNode for ALTER TABLE DROP COLUMN statement.
 * @param pos The parser position.
 * @param ifExists If the IF EXISTS clause is present. If true, will not error even if the table does not exist.
 * @param table The table to drop the columns from.
 * @param dropCols The SqlNodeList containing CompoundIdentifiers of the columns to be dropped.
 * @param ifColumnExists If the IF EXISTS clause for the column is present. If true, will not error even if columns
 *                       in dropCols do not exist.
 */
class SqlAlterTableDropCol(
    pos: SqlParserPos,
    ifExists: Boolean,
    table: SqlIdentifier,
    val dropCols: SqlNodeList,
    val ifColumnExists: Boolean,
) : SqlAlterTable(pos, ifExists, table) {
    override fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("DROP")
        writer.keyword("COLUMN")
        if (ifColumnExists) {
            writer.keyword("IF")
            writer.keyword("EXISTS")
        }
        dropCols.unparse(writer, leftPrec, rightPrec)
    }

    override fun getOperandList(): List<SqlNode> = listOf(table, dropCols)
}
