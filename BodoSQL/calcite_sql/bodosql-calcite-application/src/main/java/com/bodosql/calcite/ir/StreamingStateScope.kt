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

    private var operators: HashMap<Int, Pair<OperatorPipelineRange, OperatorType>> = HashMap()

    fun hasOperators(): Boolean {
        return operators.isNotEmpty()
    }

    fun startOperator(
        opID: Int,
        startPipelineID: Int,
        type: OperatorType,
        memoryEstimate: Int = -1,
    ) {
        if (operators.containsKey(opID)) {
            throw Exception("StreamingStateScope: Repeated Operator ID Found")
        }
        operators[opID] = Pair(OperatorPipelineRange(opID, startPipelineID), type)
        operators[opID]!!.first.memEstimate = memoryEstimate
    }

    fun endOperator(
        opID: Int,
        endPipelineID: Int,
    ) {
        if (!operators.containsKey(opID)) {
            throw Exception("StreamingStateScope: Ending Pipeline Range of Unknown Operator")
        }
        if (operators[opID]?.first?.endPipelineID != null) {
            throw Exception("StreamingStateScope: Repeated Operator End Found")
        }
        if (endPipelineID < operators[opID]!!.first.startPipelineID) {
            throw Exception("StreamingStateScope: endPipelineID cannot be less than startPipelineID")
        }
        operators[opID]!!.first.endPipelineID = endPipelineID
    }

    fun genOpComptrollerInit(): List<Op> {
        val inits =
            mutableListOf<Op>(
                Op.Stmt(Expr.Call("bodo.libs.memory_budget.init_operator_comptroller")),
            )

        for ((_, op) in operators) {
            val range = op.first
            val type = op.second
            inits.add(
                Op.Stmt(
                    Expr.Call(
                        "bodo.libs.memory_budget.register_operator",
                        Expr.IntegerLiteral(range.opID),
                        Expr.Attribute(Expr.Raw("bodo.libs.memory_budget.OperatorType"), type.toString()),
                        Expr.IntegerLiteral(range.startPipelineID),
                        Expr.IntegerLiteral(
                            range.endPipelineID ?: throw Exception(
                                "StreamingStateScope: An Operator's End Pipeline" +
                                    " Not Set Before Generating Code",
                            ),
                        ),
                        Expr.IntegerLiteral(range.memEstimate),
                    ),
                ),
            )
        }

        inits.add(Op.Stmt(Expr.Call("bodo.libs.memory_budget.compute_satisfiable_budgets")))

        return inits
    }

    fun genQueryProfileCollectorInit(): List<Op> {
        return listOf(Op.Stmt(Expr.Call("bodo.libs.query_profile_collector.init")))
    }

    fun genQueryProfileCollectorFinalize(): Op {
        return Op.Stmt(Expr.Call("bodo.libs.query_profile_collector.finalize"))
    }

    fun addToFrame(frame: Frame) {
        if (operators.size > 0) {
            // This order is important. Because we're prepending, this ensures that all OperatorComptroller calls come before the QueryProfileCollector calls, which allows the QueryProfiler to read the initial allocated budget.
            frame.prependAll(genQueryProfileCollectorInit())
            frame.prependAll(genOpComptrollerInit())

            // Insert the finalize call for the QueryProfileCollector as the last step
            frame.addBeforeReturn(genQueryProfileCollectorFinalize())
        }
    }
}
