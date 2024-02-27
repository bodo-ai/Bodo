package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlCall
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.SqlSpecialOperator
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

class SqlCopyIntoLocation(
    val pos: SqlParserPos,
    // When copying into a location, the target can be either
    // an internal stage, external stage, or external location
    val targetType: CopyIntoLocationTarget,
    val target: SqlNode,
    // When copying into a location, the source can either be
    // a table, or an arbitrary query
    val sourceType: CopyIntoLocationSource,
    val sourceNode: SqlNode,
    val partition: SqlNode?,
    val fileFormat: SqlSnowflakeFileFormat?,
) : SqlCall(pos) {
    enum class CopyIntoLocationTarget { STAGE, LOCATION }

    enum class CopyIntoLocationSource { TABLE, QUERY }

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
        when (targetType) {
            CopyIntoLocationTarget.STAGE -> {
                writer.literal(target.toString())
            }
            CopyIntoLocationTarget.LOCATION -> {
                target.unparse(writer, leftPrec, rightPrec)
            }
        }
        writer.keyword("FROM")
        when (sourceType) {
            CopyIntoLocationSource.TABLE -> {
                sourceNode.unparse(writer, leftPrec, rightPrec)
            }
            CopyIntoLocationSource.QUERY -> {
                writer.newlineAndIndent()
                writer.keyword("(")
                sourceNode.unparse(writer, leftPrec, rightPrec)
                writer.keyword(")")
            }
        }
        partition?.let {
            writer.keyword("PARTITION BY")
            partition.unparse(writer, leftPrec, rightPrec)
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
