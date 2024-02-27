package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.ddl.SqlCreateTable
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * Abstract base class to describe any `CREATE TABLE` statement
 */
abstract class SqlSnowflakeCreateTableBase(
    pos: SqlParserPos?,
    replace: Boolean,
    tableType: CreateTableType?,
    ifNotExists: Boolean,
    name: SqlIdentifier?,
    columnList: SqlNodeList?,
    query: SqlNode?,
    private val tableCommentNode: SqlNode?,
) : SqlCreateTable(pos, replace, ifNotExists, name, columnList, query) {
    val meta = SnowflakeCreateTableMetadata()

    init {
        createType = tableType
        tableCommentNode?.let { meta.setTableComment(it) }
        columnList?.let { meta.setColumnComments(it) }
    }

    override fun unparse(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("CREATE")
        if (replace) {
            writer.keyword("OR REPLACE")
        }
        if (!createTableType.equals(CreateTableType.DEFAULT)) {
            writer.keyword(createTableType.asStringKeyword())
        }
        writer.keyword("TABLE")
        if (ifNotExists) {
            writer.keyword("IF NOT EXISTS")
        }
        name.unparse(writer, leftPrec, rightPrec)
        getcolumnList()?.let {
            val frame = writer.startList("(", ")")
            for (c in it) {
                writer.sep(",")
                c.unparse(writer, 0, 0)
            }
            writer.endList(frame)
        }
        tableCommentNode?.let {
            writer.keyword("COMMENT")
            writer.keyword("=")
            it.unparse(writer, leftPrec, rightPrec)
        }
        unparseSuffix(writer, leftPrec, rightPrec)
    }

    open fun getColumnCommentStrings(): List<String?>? = null

    abstract fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    )
}
