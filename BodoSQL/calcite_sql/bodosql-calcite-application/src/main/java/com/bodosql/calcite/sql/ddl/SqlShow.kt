package com.bodosql.calcite.sql.ddl

import org.apache.calcite.sql.SqlCall
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos

/**
 * Base class for SHOW statements parse tree nodes. The portion of the statement covered by this
 * class is simply the keyword "SHOW". Subclasses handle whatever comes after.
 */
abstract class SqlShow protected constructor(
    pos: SqlParserPos,
    val isTerse: Boolean,
) : SqlCall(pos) {
    override fun unparse(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.keyword("SHOW")
        if (this.isTerse) {
            writer.keyword("TERSE")
        }
        unparseShowOperation(writer, leftPrec, rightPrec)
    }

    /** This unparses the clause after SHOW.  */
    protected abstract fun unparseShowOperation(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    )
}
