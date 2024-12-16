package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlBasicCall
import org.apache.calcite.sql.SqlCall
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.SqlSelect
import org.apache.calcite.sql.SqlSpecialOperator
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

class SqlCopyIntoTable(
    val pos: SqlParserPos,
    val target: SqlNode,
    val targetCols: List<SqlNode>?,
    // When copying into a table, the source can either be
    // an internal stage, external stage, external location
    // or a transformation query on one of the above
    val sourceType: CopyIntoTableSource,
    val sourceNode: SqlNode,
    val pattern: SqlNode?,
    val fileFormat: SqlSnowflakeFileFormat?,
) : SqlCall(pos) {
    enum class CopyIntoTableSource { LOCATION, STAGE, QUERY }

    companion object {
        @JvmField
        val OPERATOR = SqlSpecialOperator("COPY INTO", SqlKind.OTHER_DDL)
    }

    override fun unparse(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("COPY INTO")
        target.unparse(writer, leftPrec, rightPrec)
        targetCols?.let {
            val frame = writer.startList("(", ")")
            for (c in targetCols) {
                writer.sep(",")
                c.unparse(writer, 0, 0)
            }
            writer.endList(frame)
        }
        writer.keyword("FROM")
        when (sourceType) {
            CopyIntoTableSource.LOCATION -> {
                sourceNode.unparse(writer, leftPrec, rightPrec)
            }
            CopyIntoTableSource.STAGE -> {
                writer.literal(sourceNode.toString())
            }
            CopyIntoTableSource.QUERY -> {
                writer.newlineAndIndent()
                writer.keyword("(")
                writer.keyword("SELECT")
                val sourceSelect = (sourceNode as SqlSelect)
                writer.list(
                    SqlWriter.FrameTypeEnum.SELECT_LIST,
                    SqlWriter.COMMA,
                    sourceSelect.selectList,
                )
                writer.newlineAndIndent()
                writer.keyword("FROM")
                val from = sourceSelect.from
                if (from is SqlBasicCall) {
                    assert(from.operator.kind == SqlKind.AS)
                    writer.literal(from.operandList[0].toString())
                    writer.keyword("AS")
                    from.operandList[1].unparse(writer, leftPrec, rightPrec)
                } else {
                    writer.literal(from.toString())
                }
                writer.keyword(")")
            }
        }
        pattern?.let {
            writer.keyword("PATTERN")
            writer.keyword("=")
            pattern.unparse(writer, leftPrec, rightPrec)
        }
        fileFormat?.let {
            writer.keyword("FILE_FORMAT")
            writer.keyword("=")
            fileFormat.unparse(writer, leftPrec, rightPrec)
        }
    }

    override fun getOperator(): SqlOperator = OPERATOR

    override fun getOperandList(): List<SqlNode> = listOf(target, sourceNode)
}
