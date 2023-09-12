package com.bodosql.calcite.ir

import java.lang.Exception

class StreamingStateScope {

    /**
     * Helper Class to Store Operator Info and Ranges
     */
    private class OperatorPipelineRange(val opID: Int, val startPipelineID: Int, val opType: Int? = null) {
        var endPipelineID: Int? = null
        var memEstimate: Int = -1
    }

    private var operators: HashMap<Int, OperatorPipelineRange> = HashMap()

    fun startOperator(opID: Int, startPipelineID: Int) {
        if (operators.containsKey(opID)) {
            throw Exception("StreamingStateScope: Repeated Operator ID Found")
        }
        operators[opID] = OperatorPipelineRange(opID, startPipelineID)
    }

    fun endOperator(opID: Int, endPipelineID: Int) {
        if (!operators.containsKey(opID)) {
            throw Exception("StreamingStateScope: Ending Pipeline Range of Unknown Operator")
        }
        if (operators[opID]?.endPipelineID != null) {
            throw Exception("StreamingStateScope: Repeated Operator End Found")
        }
        operators[opID]!!.endPipelineID = endPipelineID
    }

    fun genOpComptrollerInit(): List<Op> {
        val inits = mutableListOf<Op>(
            Op.Assign(Variable("comptroller"), Expr.Call("operator_comptroller_init")),
        )

        for ((_, op) in operators) {
            inits.add(
                Op.Stmt(
                    Expr.Call(
                        "operator_comptroller_register",
                        Expr.IntegerLiteral(op.opID),
                        Expr.IntegerLiteral(op.startPipelineID),
                        Expr.IntegerLiteral(op.endPipelineID ?: throw Exception("StreamingStateScope: An Operator's End Pipeline Not Set Before Generating Code")),
                        Expr.IntegerLiteral(op.memEstimate),
                    ),
                ),
            )
        }

        return inits
    }

    fun genOpComptrollerFinalize() {}
}
