package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * Object to describe a `CREATE TABLE` statement using `AS SELECT`
 */
class BodoSqlCreateTableAs(
    pos: SqlParserPos?,
    replace: Boolean,
    tableType: CreateTableType?,
    ifNotExists: Boolean,
    name: SqlIdentifier?,
    columnList: SqlNodeList?,
    val selectSource: SqlNode,
    val clusterExprs: SqlNodeList?,
    val copyGrants: Boolean,
    comment: SqlNode?,
    keyList: SqlNodeList?,
    valuesList: SqlNodeList?,
) : BodoSqlCreateTableBase(pos, replace, tableType, ifNotExists, name, columnList, selectSource, comment, keyList, valuesList) {
    private val columnComments: List<SqlNode?>?

    init {
        // CREATE TABLE AS SELECT only allows COPY GRANTS if OR REPLACE is also provided
        if (copyGrants && !replace) {
            throw Exception("CREATE TABLE with AS SELECT requires OR REPLACE to use COPY GRANTS")
        }
        columnComments = columnList?.list?.map { (it as SqlSnowflakeColumnDeclaration).comment }
    }

    override fun getColumnCommentStrings(): List<String?>? =
        getcolumnList()?.map {
            (it as SqlSnowflakeColumnDeclaration).comment?.let {
                    c ->
                (c as SqlLiteral).getValueAs(String::class.java)
            }
        }

    override fun unparseSuffix(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        clusterExprs?.let {
            writer.keyword("CLUSTER BY")
            val frame = writer.startList("(", ")")
            clusterExprs.unparse(writer, leftPrec, rightPrec)
            writer.endList(frame)
        }
        if (copyGrants) writer.keyword("COPY GRANTS")
        writer.keyword("AS")
        writer.newlineAndIndent()
        selectSource.unparse(writer, 0, 0)
    }
}
