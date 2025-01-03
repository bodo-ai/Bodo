package com.bodosql.calcite.application

import com.bodosql.calcite.table.ColumnDataTypeInfo

/**
 * This class is the entry point for all Python code that relates to planner driven operations.
 * Each method in this class should be a static method so that it can be cleanly called from Python
 * without relying on any method behavior.
 *
 * The motivation behind this is three-fold:
 * 1. By adding an explicit Python interface it allows common IDE
 * operations (e.g. dead code detection) to be performed on all the
 * "core" files except for the Python entry point files.
 * 2. It ensures that any of the internal functionality can be defined as
 * "package private". Anything exposed to Python must be public.
 * 3. It defines an explicit interface between Python and Java, rather than
 * depending on Py4J's support for Java method calls. This means if we ever
 * want to switch a proper "service", then these APIs are much easier to
 * support.
 */
class PythonEntryPoint {
    companion object {
        /**
         * Parse a query and update the generator's state.
         * @param generator The generator to update.
         * @param query The query to parse.
         */
        @JvmStatic
        fun parseQuery(
            generator: RelationalAlgebraGenerator,
            query: String,
        ) {
            generator.parseQuery(query)
        }

        /**
         * Reset the planner inside the RelationalAlgebraGenerator.
         * @param generator The generator to reset.
         */
        @JvmStatic
        fun resetPlanner(generator: RelationalAlgebraGenerator) {
            generator.reset()
        }

        /**
         * Generate a string representation of the optimized plan.
         * @param generator The generator to use.
         * @param sql The SQL query to optimize.
         * @param includeCosts Whether to include costs in the output plan.
         * @return The string representation of the optimized plan.
         */
        @JvmStatic
        fun getOptimizedPlanString(
            generator: RelationalAlgebraGenerator,
            sql: String,
            includeCosts: Boolean,
        ): String =
            generator.getOptimizedPlanString(
                sql,
                includeCosts,
                listOf(),
                mapOf(),
            )

        /**
         * Generate a string representation of the generated code and the corresponding
         * optimized plan.
         * @param generator The generator to use.
         * @param sql The SQL query to optimize.
         * @param includeCosts Whether to include costs in the output plan.
         * @return The generated code and the string representation of
         * the optimized plan.
         */
        @JvmStatic
        fun getPandasAndPlanString(
            generator: RelationalAlgebraGenerator,
            sql: String,
            includeCosts: Boolean,
        ): PandasCodeSqlPlanPair = generator.getPandasAndPlanString(sql, includeCosts, listOf(), mapOf())

        /**
         * Generate the Python code to execute the given SQL query.
         * @param generator The generator to use.
         * @param sql The SQL query to optimize.
         * @param dynamicParamTypes The dynamic parameter types.
         * @param namedParamTypeMap The named parameter types.
         * @return The generated code.
         */
        @JvmStatic
        fun getPandasString(
            generator: RelationalAlgebraGenerator,
            sql: String,
            dynamicParamTypes: MutableList<ColumnDataTypeInfo>,
            namedParamTypeMap: MutableMap<String, ColumnDataTypeInfo>,
        ): String = generator.getPandasString(sql, dynamicParamTypes, namedParamTypeMap)
    }
}
