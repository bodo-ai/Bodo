package com.bodosql.calcite.adapter.bodo.window

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Variable

/**
 * Metadata about the initialization of the generated apply function.
 */
internal data class WindowApplyFuncHeader(
    /**
     * Input dataframe passed into the function.
     */
    val input: BodoEngineTable,
    /**
     * A variable that references the original index.
     */
    val index: Variable,
    /**
     * A variable that references the length of the dataframe.
     */
    val len: Variable,
    /**
     * Column index for the column with the original positioning information.
     */
    val position: Int,
)
