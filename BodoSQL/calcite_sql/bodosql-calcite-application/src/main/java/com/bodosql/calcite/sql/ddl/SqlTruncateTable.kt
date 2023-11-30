package com.bodosql.calcite.sql.ddl

import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.SqlDdl
import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.SqlSpecialOperator
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

class SqlTruncateTable(pos: SqlParserPos, val ifExists: Boolean, val name: SqlIdentifier) : SqlDdl(OPERATOR, pos) {
    companion object {
        @JvmStatic
        val OPERATOR: SqlOperator =
            SqlSpecialOperator("TRUNCATE TABLE", SqlKind.TRUNCATE_TABLE)
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
