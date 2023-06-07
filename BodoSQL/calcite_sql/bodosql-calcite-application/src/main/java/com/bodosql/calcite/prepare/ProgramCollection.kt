package com.bodosql.calcite.prepare

import org.apache.calcite.tools.Program

/**
 * Collection of programs each planner must define.
 */
data class ProgramCollection(val preprocessor: Program, val unoptimized: Program, val optimized: Program) {
    /**
     * Structures the above into a list for the planner.
     *
     * The list order is preprocessor, optimized, and then unoptimized.
     */
    fun toList(): List<Program> = listOf(preprocessor, optimized, unoptimized)
}
