package com.bodosql.calcite.sql.ddl

import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.*
import org.apache.calcite.sql.parser.SqlParserPos

class SqlTruncateTable(pos: SqlParserPos, val ifExists: Boolean, val name: SqlIdentifier) : SqlDdl(OPERATOR, pos) {
    companion object {
        @JvmStatic
        val OPERATOR: SqlOperator =
            SqlSpecialOperator("TRUNCATE TABLE", SqlKind.OTHER_DDL)
    }

    override fun getOperandList(): List<SqlNode> = ImmutableList.of(name)

    override fun unparse(writer: SqlWriter, leftPrec: Int, rightPrec: Int) {
        writer.keyword(operator.name)
        if (ifExists) {
            writer.keyword("IF EXISTS")
        }
        name.unparse(writer, leftPrec, rightPrec)
    }
}
