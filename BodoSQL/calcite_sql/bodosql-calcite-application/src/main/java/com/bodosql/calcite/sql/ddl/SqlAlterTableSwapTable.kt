package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.*
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.util.ImmutableNullableList

class SqlAlterTableSwapTable(
    pos: SqlParserPos,
    ifExists : Boolean,
    table : SqlIdentifier,
    val swapTable : SqlIdentifier,
) : SqlAlterTable(pos, ifExists, table) {

    override fun unparseSuffix(writer: SqlWriter, leftPrec: Int, rightPrec: Int) {
        writer.keyword("SWAP")
        writer.keyword("WITH")
        swapTable.unparse(writer, leftPrec, rightPrec)
    }

    override fun getOperandList(): List<SqlNode> = listOf(table, swapTable)
}
