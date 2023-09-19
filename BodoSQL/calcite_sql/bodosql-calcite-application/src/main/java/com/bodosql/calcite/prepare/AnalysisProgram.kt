package com.bodosql.calcite.prepare

import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.tools.Program

/**
 * A special type of program that returns the input RelNode unmodified, but
 * first performs some type of analysis on it that can later be retrieved.
 *
 * Also stores the input RelNode plan before doing the analysis.
 *
 * The analysis is not done if the environment variable ANALYZE_PLAN is provided
 * and non-zero.
 */
abstract class AnalysisProgram : Program {
    private var inputPlan: RelNode? = null

    fun retrieveInputPlan(): RelNode {
        if (!AnalysisSuite.enableAnalysis) throw Exception("Cannot call retrieveInputPlan unless AnalysisSuite.enableAnalysis is set")
        return inputPlan ?: throw Exception("Cannot call retrieveInputPlan before an analysis program has been run")
    }

    /**
     * Resets the analyzer state and any variables defined
     * by the subclass.
     */
    abstract fun reset()

    /**
     * Runs the desired analysis and stores the outcome in
     * any variables defined by the subclass.
     */
    abstract fun runAnalysis()

    override fun run(
        planner: RelOptPlanner,
        rel: RelNode,
        requiredOutputTraits: RelTraitSet,
        materializations: MutableList<RelOptMaterialization>,
        lattices: MutableList<RelOptLattice>,
    ): RelNode {
        if (AnalysisSuite.enableAnalysis) {
            reset()
            inputPlan = rel
            runAnalysis()
        }
        return rel
    }
}
