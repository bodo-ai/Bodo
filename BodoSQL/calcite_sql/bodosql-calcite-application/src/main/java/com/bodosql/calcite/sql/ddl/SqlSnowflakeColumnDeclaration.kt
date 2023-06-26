package com.bodosql.calcite.sql.ddl

import org.apache.calcite.schema.ColumnStrategy
import org.apache.calcite.sql.*
import org.apache.calcite.sql.ddl.SqlColumnDeclaration
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.util.Pair;

/**
 * Subclass of SqlColumnDeclaration which includes additional qualifiers to a column
 */
class SqlSnowflakeColumnDeclaration(
    pos: SqlParserPos?,
    name: SqlIdentifier,
    dataType: SqlDataTypeSpec,
    val defaultExpr: SqlNode?,
    val incrementExpr: Pair<SqlLiteral,SqlLiteral>?,
    strategy: ColumnStrategy?,
) : SqlColumnDeclaration(pos, name, dataType, defaultExpr, strategy) {

    override fun unparse(writer: SqlWriter, leftPrec: Int, rightPrec: Int) {
        name.unparse(writer, 0, 0)
        dataType.unparse(writer, 0, 0)
        dataType.nullable?.let {
            if (!it) {
                writer.keyword("NOT NULL")
            }
        }
        defaultExpr?.let {
            writer.keyword("DEFAULT")
            defaultExpr.unparse(writer, leftPrec, rightPrec)
        }
        incrementExpr?.let {
            writer.keyword("AUTOINCREMENT")
            writer.sep("(")
            incrementExpr.left.unparse(writer, leftPrec, rightPrec)
            writer.sep(",")
            incrementExpr.right.unparse(writer, leftPrec, rightPrec)
            writer.sep(")")
        }
    }

    companion object {
        private val OPERATOR = SqlSpecialOperator("COLUMN_DECL", SqlKind.COLUMN_DECL)
    }
}
