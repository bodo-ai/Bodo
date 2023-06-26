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
 * Object to describe a `CREATE TABLE` statement using `CLONE`
 */
class SqlSnowflakeCreateTableClone(
    pos: SqlParserPos?,
    replace: Boolean,
    tableType: CreateTableType?,
    ifNotExists: Boolean,
    name: SqlIdentifier?,
    val cloneSource: SqlNode,
    val copyGrants : Boolean,
) : SqlSnowflakeCreateTableBase(pos, replace, tableType, ifNotExists, name, null, cloneSource) {

    override fun unparseSuffix(writer: SqlWriter, leftPrec: Int, rightPrec: Int) {
        writer.keyword("CLONE")
        cloneSource.unparse(writer, 0, 0)
        if (copyGrants) writer.keyword("COPY GRANTS")
    }
}
