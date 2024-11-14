package com.bodosql.calcite.sql.func

import org.apache.calcite.sql.SqlDynamicParam
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.util.Litmus

class SqlNamedParam(
    baseParamName: String,
    pos: SqlParserPos,
) : SqlDynamicParam(-1, pos) {
    // Remove the leading @ from the parameter name.
    // TODO: Handle in parser.
    val paramName = baseParamName.substring(1)

    // ~ Methods ----------------------------------------------------------------
    override fun clone(pos: SqlParserPos): SqlNode {
        // TODO: Remove @ and handle in parser
        return SqlNamedParam("@$paramName", pos)
    }

    override fun getIndex(): Int {
        // TODO: Will this be an issue with the SqlDynamicParam equality check?
        throw UnsupportedOperationException("Named parameters do not have an index")
    }

    override fun unparse(
        writer: SqlWriter,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        writer.print("@$paramName")
    }

    override fun equalsDeep(
        node: SqlNode?,
        litmus: Litmus,
    ): Boolean {
        if (node !is SqlNamedParam) {
            return litmus.fail("{} != {}", this, node)
        }
        return if (paramName != node.paramName) {
            litmus.fail("{} != {}", this, node)
        } else {
            litmus.succeed()
        }
    }
}
