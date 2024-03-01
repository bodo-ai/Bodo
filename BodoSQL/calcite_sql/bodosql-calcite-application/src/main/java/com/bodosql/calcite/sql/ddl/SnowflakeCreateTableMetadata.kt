package com.bodosql.calcite.sql.ddl

import com.bodosql.calcite.ir.Expr
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.type.SqlTypeName

class SnowflakeCreateTableMetadata() {
    var tableComment: String? = null
    var columnComments: List<String?>? = null

    fun setTableComment(commentNode: SqlNode) {
        if (commentNode is SqlLiteral && commentNode.typeName == SqlTypeName.CHAR) {
            tableComment = commentNode.getValueAs(String::class.java)
        } else {
            throw Exception("Invalid table comment: $commentNode")
        }
    }

    fun setColumnComments(commentNode: SqlNodeList) {
        columnComments =
            commentNode.list.map {
                if (it is SqlSnowflakeColumnDeclaration) {
                    if (it.comment is SqlLiteral && it.comment.typeName == SqlTypeName.CHAR) {
                        it.comment.getValueAs(String::class.java)
                    } else if (it.comment != null) {
                        throw Exception("Invalid column declaration: ${it.comment}")
                    } else {
                        null
                    }
                } else if (it != null) {
                    throw Exception("Invalid column declaration: $it")
                } else {
                    null
                }
            }
    }

    fun emitCtasExpr(): Expr {
        val ctasMetaKwargs: MutableList<Pair<String, Expr>> = ArrayList()
        if (tableComment != null) {
            val tableCommentExpr: Expr = Expr.StringLiteral(tableComment!!)
            ctasMetaKwargs.add(Pair("table_comments", tableCommentExpr))
        }
        if (columnComments != null) {
            val columnCommentExprs: MutableList<Expr> = ArrayList()
            for (columnComment in columnComments!!) {
                if (columnComment == null) {
                    columnCommentExprs.add(Expr.None)
                } else {
                    columnCommentExprs.add(Expr.StringLiteral(columnComment))
                }
            }
            val columnCommentTuple: Expr = Expr.Tuple(columnCommentExprs)
            ctasMetaKwargs.add(Pair("column_comments", columnCommentTuple))
        }
        return Expr.Call("bodo.utils.typing.SnowflakeCreateTableMetaType", listOf(), ctasMetaKwargs)
    }
}
