package com.bodosql.calcite.prepare

/**
 * A collection of AnalysisProgram objects that can be used to collect data about
 * the plan at various stages of the optimizer.
 *
 * To add a new analysis pass, create a new subclass of AnalysisProgram then
 * add it as a field in the companion object below. It can be accessed by
 * BodoPrograms.kt or testing classes.
 *
 */
class AnalysisSuite {
    companion object {
        @JvmField
        var enableAnalysis = false

        val multiJoinAnalyzer = MultiJoinAnalyzerProgram()
        val filterPushdownAnalysis = FilterPushdownAnalyzerProgram()
    }
}
