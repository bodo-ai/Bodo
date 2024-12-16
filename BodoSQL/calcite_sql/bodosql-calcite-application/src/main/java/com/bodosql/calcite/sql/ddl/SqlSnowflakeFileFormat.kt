package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlWriter
import java.util.HashMap

class SqlSnowflakeFileFormat(
    val formatName: SqlNode?,
    val formatType: SqlNode?,
    val formatOptions: HashMap<String, String>,
) {
    fun unparse(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("(")
        formatName?.let {
            writer.keyword("FORMAT_NAME")
            writer.keyword("=")
            formatName.unparse(writer, leftPrec, rightPrec)
        }
        formatType?.let {
            writer.keyword("TYPE")
            writer.keyword("=")
            formatType.unparse(writer, leftPrec, rightPrec)
            val frame = writer.startList("", "")
            for (c in formatOptions.toSortedMap()) {
                writer.sep(",")
                writer.keyword(String.format("%s = %s", c.key, c.value))
            }
            writer.endList(frame)
        }
        writer.keyword(")")
    }
}
