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

/**
 * SQL Node for DESCRIBE VIEW <name>
 */
class SqlDescribeView(
    pos: SqlParserPos,
    val view: SqlIdentifier,
) : SqlCall(pos) {
    companion object {
        @JvmStatic
        private val OPERATOR: SqlOperator =
            SqlSpecialOperator("DESCRIBE VIEW", SqlKind.DESCRIBE_VIEW)
    }

    override fun getOperandList(): List<SqlNode> = ImmutableList.of(view)

    override fun getOperator(): SqlOperator = OPERATOR

    override fun unparse(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword(operator.name)
        view.unparse(writer, leftPrec, rightPrec)
    }
}
