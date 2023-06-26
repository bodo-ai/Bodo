package com.bodosql.calcite.sql.ddl

import com.bodosql.calcite.sql.validate.BodoSqlValidator
import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.ddl.SqlCreateTable
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.sql.validate.SqlValidator
import org.apache.calcite.sql.validate.SqlValidatorScope

/**
 * Object to describe a `CREATE TABLE` statement using `LIKE`
 */
class SqlSnowflakeCreateTableLike(
    pos: SqlParserPos?,
    replace: Boolean,
    tableType: CreateTableType?,
    ifNotExists: Boolean,
    name: SqlIdentifier?,
    val likeSource: SqlNode,
    val clusterExprs : SqlNodeList?,
    val copyGrants : Boolean,
) : SqlSnowflakeCreateTableBase(pos, replace, tableType, ifNotExists, name, null, likeSource) {

    override fun unparseSuffix(writer: SqlWriter, leftPrec: Int, rightPrec: Int) {
        writer.keyword("LIKE")
        likeSource.unparse(writer, 0, 0)
        clusterExprs?.let{
            writer.keyword("CLUSTER BY")
            val frame = writer.startList("(", ")")
            clusterExprs.unparse(writer, leftPrec, rightPrec)
            writer.endList(frame)
        }
        if (copyGrants) writer.keyword("COPY GRANTS")
    }
}
