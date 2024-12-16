package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlSelect
import org.apache.calcite.sql.SqlUpdate
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.util.Pair

class SqlSnowflakeUpdate(
    pos: SqlParserPos,
    targetTable: SqlNode,
    targetColumnList: SqlNodeList,
    sourceExpressionList: SqlNodeList,
    condition: SqlNode?,
    sourceSelect: SqlSelect?,
    alias: SqlIdentifier?,
    val from: SqlNode?,
) : SqlUpdate(pos, targetTable, targetColumnList, sourceExpressionList, condition, sourceSelect, alias) {
    override fun unparse(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        val frame = writer.startList(SqlWriter.FrameTypeEnum.SELECT, "UPDATE", "")
        val opLeft = operator.leftPrec
        val opRight = operator.rightPrec
        targetTable.unparse(writer, opLeft, opRight)
        val alias = alias
        alias?.let {
            writer.keyword("AS")
            alias.unparse(writer, opLeft, opRight)
        }
        val setFrame = writer.startList(SqlWriter.FrameTypeEnum.UPDATE_SET_LIST, "SET", "")
        for (pair in Pair.zip(targetColumnList, sourceExpressionList)) {
            writer.sep(",")
            val id = pair.left as SqlIdentifier
            id.unparse(writer, opLeft, opRight)
            writer.keyword("=")
            val sourceExp = pair.right
            sourceExp.unparse(writer, opLeft, opRight)
        }
        writer.endList(setFrame)
        // Cannot invoke superclass' version of this method because
        // the FROM clause is interleaved in the middle
        from?.let {
            writer.sep("FROM")
            from.unparse(writer, opLeft, opRight)
        }
        val condition = condition
        if (condition != null) {
            writer.sep("WHERE")
            condition.unparse(writer, opLeft, opRight)
        }
        writer.endList(frame)
    }
}
