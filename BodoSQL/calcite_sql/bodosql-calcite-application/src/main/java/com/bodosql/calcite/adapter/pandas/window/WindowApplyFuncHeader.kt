package com.bodosql.calcite.adapter.pandas.window

import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Variable

/**
 * Metadata about the initialization of the generated apply function.
 */
internal data class WindowApplyFuncHeader(
    /**
     * Input dataframe passed into the function.
     */
    val input: Dataframe,

    /**
     * A variable that references the original index.
     */
    val index: Variable,

    /**
     * A variable that references the length of the dataframe.
     */
    val len: Variable,

    /**
     * Column name for the column with the original positioning information.
     */
    val position: String,
)
