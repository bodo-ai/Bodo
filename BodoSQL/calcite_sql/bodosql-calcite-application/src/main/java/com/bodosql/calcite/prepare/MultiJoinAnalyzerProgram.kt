package com.bodosql.calcite.prepare

import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelVisitor
import org.apache.calcite.rel.rules.MultiJoin

/**
 * An AnalysisProgram that keeps track of the number of MultiJoin nodes
 * in a program and the number of inputs to each MultiJoin.
 */
class MultiJoinAnalyzerProgram : AnalysisProgram() {
    private val multiJoins: MutableList<Int> = mutableListOf()

    /**
     * Fetch the list of all MultiJoin nodes (by the # of inputs)
     * from the analyzed program.
     */
    fun getMultiJoins(): MutableList<Int> = multiJoins

    override fun reset() {
        multiJoins.clear()
    }

    /**
     * Use a RelVisitor to hunt for all MultiJoin nodes in the program. Whenever
     * a MultiJoin is found, add its length to the stored list of lengths.
     */
    override fun runAnalysis() {
        val countVisitor = MultiJoinCounter(multiJoins)
        countVisitor.go(retrieveInputPlan())
    }

    private class MultiJoinCounter(
        val multiJoinList: MutableList<Int>,
    ) : RelVisitor() {
        override fun visit(
            node: RelNode,
            ordinal: Int,
            parent: RelNode?,
        ) {
            if (node is MultiJoin) {
                multiJoinList.add(node.inputs.size)
            }
            node.childrenAccept(this)
        }
    }
}
