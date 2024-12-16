package com.bodosql.calcite.prepare

import org.apache.calcite.tools.Program

typealias ProgramFactory = (optimize: Boolean) -> Program

/**
 * Collection of programs each planner must define.
 */
data class ProgramCollection(
    val programFactory: ProgramFactory,
) {
    /**
     * Structures the above into a list for the planner.
     *
     * The list order is preprocessor, optimized, and then unoptimized.
     */
    fun toList(): List<Program> =
        listOf(
            BodoPrograms.preprocessor(),
            programFactory.invoke(true),
            programFactory.invoke(false),
        )
}
