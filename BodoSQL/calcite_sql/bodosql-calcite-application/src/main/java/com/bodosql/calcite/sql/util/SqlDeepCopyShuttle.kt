package com.bodosql.calcite.sql.util

import org.apache.calcite.sql.SqlCall
import org.apache.calcite.sql.SqlDataTypeSpec
import org.apache.calcite.sql.SqlDynamicParam
import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlIntervalQualifier
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlTableIdentifierWithID
import org.apache.calcite.sql.util.SqlShuttle

/**
 * Implementation of a SqlShuttle that creates a deep copy by
 * recursively calling clone on all nodes.
 */
class SqlDeepCopyShuttle : SqlShuttle() {
    /**
     * Basic implementation that just updates the position and
     * calls clone.
     */
    private fun defaultImpl(node: SqlNode): SqlNode = node.clone(node.parserPosition)

    // ~ Methods ----------------------------------------------------------------
    override fun visit(literal: SqlLiteral): SqlNode = defaultImpl(literal)

    override fun visit(id: SqlIdentifier): SqlNode = defaultImpl(id)

    override fun visit(id: SqlTableIdentifierWithID): SqlNode = defaultImpl(id)

    override fun visit(type: SqlDataTypeSpec): SqlNode = defaultImpl(type)

    override fun visit(param: SqlDynamicParam): SqlNode = defaultImpl(param)

    override fun visit(intervalQualifier: SqlIntervalQualifier): SqlNode = defaultImpl(intervalQualifier)

    /**
     * Note: This can't use the builtin SqlShuttle implementation
     * because that may not always return a copy.
     */
    override fun visit(call: SqlCall): SqlNode {
        val newOperandList: MutableList<SqlNode?> = java.util.ArrayList()
        // See above comment, operand list is nullable, but it isn't typed as such
        val curOperandList: List<SqlNode?> = call.operandList
        for (i in curOperandList.indices) {
            val curNode = curOperandList[i]
            if (curNode == null) {
                newOperandList.add(null)
            } else {
                newOperandList.add(curNode.accept(this))
            }
        }
        return call.operator.createCall(call.functionQuantifier, call.parserPosition, newOperandList)
    }

    /**
     * Note: This can't use the builtin SqlShuttle implementation
     * because that may not always return a copy.
     */
    override fun visit(nodeList: SqlNodeList): SqlNode {
        val origNodeList: List<SqlNode?> = nodeList.list
        val newNodeList = SqlNodeList(nodeList.parserPosition)
        for (i in origNodeList.indices) {
            val origNode = origNodeList[i]
            if (origNode == null) {
                newNodeList.add(null)
            } else {
                newNodeList.add(origNode.accept(this))
            }
        }
        return newNodeList
    }
}
