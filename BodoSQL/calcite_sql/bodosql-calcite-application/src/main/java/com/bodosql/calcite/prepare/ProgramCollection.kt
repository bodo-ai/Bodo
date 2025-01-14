package com.bodosql.calcite.prepare

import org.apache.calcite.tools.Program

typealias ProgramFactory = () -> Program

/**
 * Collection of programs each planner must define.
 */
data class ProgramCollection(
    val programFactory: ProgramFactory,
) {
    /**
     * Structures the above into a list for the planner.
     *
     * The list order is preprocessor and optimized.
     */
    fun toList(): List<Program> =
        listOf(
            BodoPrograms.preprocessor(),
            programFactory.invoke(),
        )
}
