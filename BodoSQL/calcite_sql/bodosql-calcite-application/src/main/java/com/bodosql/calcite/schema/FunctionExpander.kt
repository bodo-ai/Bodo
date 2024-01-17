package com.bodosql.calcite.schema

import com.google.common.collect.ImmutableList

/**
 * Interface that defines the API used to "inline" or "expand"
 * the contents of a function body. This is intended to be used
 * for inlining UDFs that are defined with a known function body.
 */
interface FunctionExpander {

    /**
     * Inline the body of a function. This API is responsible for
     * parsing the function body as a query and validating the contents
     * query for type stability.
     *
     * This API is still under active development, so the return type is not yet finalized
     * and additional arguments are likely to be added.
     *
     * @param functionBody Body of the function.
     * @param functionPath Path of the function.
     */
    fun expandFunction(functionBody: String, functionPath: ImmutableList<String>)
}
