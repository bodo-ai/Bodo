package com.bodosql.calcite.prepare

import com.google.common.collect.ImmutableList
import org.apache.calcite.tools.Programs

/**
 * An AnalysisProgram that checks that no filters can be pushed down at the time
 * of analysis.
 */
class FilterPushdownAnalyzerProgram : AnalysisProgram() {
    private var canPushdownFilter = false

    fun getCanPushdownFilter() = canPushdownFilter

    override fun reset() {
        canPushdownFilter = false
    }

    override fun runAnalysis() {
        val plan = retrieveInputPlan()
        // TODO(aneesh) investigate other rules that could be relevant here
        val program = Programs.ofRules(BodoRules.PANDAS_FILTER_INTO_JOIN_RULE)
        val new_plan = program.run(plan.cluster.planner, plan, plan.traitSet, ImmutableList.of(), ImmutableList.of())
        if (!new_plan.deepEquals(plan)) {
            canPushdownFilter = true
        }
    }
}
