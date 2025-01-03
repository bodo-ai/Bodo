package com.bodosql.calcite.application

import com.bodosql.calcite.application.PythonLoggers.toggleLoggers
import com.bodosql.calcite.application.write.WriteTarget
import com.bodosql.calcite.ddl.DDLExecutionResult
import com.bodosql.calcite.table.BodoSQLColumn
import com.bodosql.calcite.table.ColumnDataTypeInfo
import org.apache.commons.lang3.exception.ExceptionUtils
import java.util.Properties

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
         * @param dynamicParamTypes The dynamic parameter types.
         * @param namedParamTypeMap The named parameter types.
         * @return The generated code and the string representation of
         * the optimized plan.
         */
        @JvmStatic
        fun getPandasAndPlanString(
            generator: RelationalAlgebraGenerator,
            sql: String,
            includeCosts: Boolean,
            dynamicParamTypes: MutableList<ColumnDataTypeInfo>,
            namedParamTypeMap: MutableMap<String, ColumnDataTypeInfo>,
        ): PandasCodeSqlPlanPair = generator.getPandasAndPlanString(sql, includeCosts, dynamicParamTypes, namedParamTypeMap)

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

        /**
         * Determine the "type" of write produced by this SQL code.
         * The write operation is always assumed to be the top level
         * of the parsed query. It returns the name of operation in
         * question to enable passing the correct write API to the table.
         *
         * Currently supported write types: "MERGE": Merge into "INSERT": Insert Into
         *
         * TODO: Remove once we refactor the MERGE INTO code for Iceberg.
         *
         * @param generator The generator to use.
         * @param sql The SQL query to parse.
         * @return A string representing the type of write.
         */
        @JvmStatic
        fun getWriteType(
            generator: RelationalAlgebraGenerator,
            sql: String,
        ): String = generator.getWriteType(sql)

        /**
         * Execute the given DDL statement in an interpreted manner. This
         * assumes/requires that sql is a DDL statement, which should have
         * already been checked.
         * @param generator The generator to use.
         * @param sql The DDL statement to execute
         * @return The result of the DDL execution.
         */
        @JvmStatic
        fun executeDDL(
            generator: RelationalAlgebraGenerator,
            sql: String,
        ): DDLExecutionResult = generator.executeDDL(sql)

        /**
         * Determine if the active query is a DDL query that is not treated like compute (not CTAS).
         * @param generator The generator to use.
         * @return Is the query DDL?
         */
        @JvmStatic
        fun isDDLProcessedQuery(generator: RelationalAlgebraGenerator): Boolean = generator.isDDLProcessedQuery

        /**
         * Build a BodoSQLColumnDataType from a type ID.
         * @param typeID The type ID to convert.
         * @return The BodoSQLColumnDataType.
         */
        @JvmStatic
        fun buildBodoSQLColumnDataTypeFromTypeId(typeID: Int): BodoSQLColumn.BodoSQLColumnDataType =
            BodoSQLColumn.BodoSQLColumnDataType.fromTypeId(typeID)

        /**
         * Build a WriteTargetEnum from a string.
         * @param value The string to convert.
         * @return The WriteTargetEnum.
         */
        @JvmStatic
        fun buildWriteTargetEnumFromString(value: String): WriteTarget.WriteTargetEnum = WriteTarget.WriteTargetEnum.fromString(value)

        /**
         * Configure the Java logging level.
         * @param level The logging level to set.
         */
        @JvmStatic
        fun configureJavaLogging(level: Int) {
            toggleLoggers(level)
        }

        /**
         * Build an ArrayList that can be transferred to Python.
         * This is done because lists are not automatically supported in Py4j.
         * @return The ArrayList.
         */
        @JvmStatic
        fun buildArrayList(): ArrayList<Any> = ArrayList()

        /**
         * Append an element to the array list. This is done to make clear
         * that the list is modified by calling into Java.
         * @param lst The list to append to.
         * @param elem The element to append.
         */
        @JvmStatic
        fun appendToArrayList(
            lst: ArrayList<Any>,
            elem: Any,
        ) {
            lst.add(elem)
        }

        /**
         * Build a map that can be transferred to Python.
         * This is done because maps are not automatically supported in Py4j.
         * @return The map.
         */
        @JvmStatic
        fun buildMap(): HashMap<Any, Any> = HashMap()

        /**
         * Put an element into the map. This is done to make clear
         * that the map is modified by calling into Java.
         * @param map The map to put into.
         * @param key The key to put.
         * @param value The value to put.
         */
        @JvmStatic
        fun putIntoMap(
            map: HashMap<Any, Any>,
            key: Any,
            value: Any,
        ) {
            map[key] = value
        }

        /**
         * Build a properties value that can be transferred to Python.
         * @return The properties value.
         */
        @JvmStatic
        fun buildProperties(): Properties = Properties()

        /**
         * Set a property in the properties object.
         * @param properties The properties object to set the property in.
         * @param key The key of the property.
         * @param value The value of the property.
         */
        @JvmStatic
        fun setProperty(
            properties: Properties,
            key: String,
            value: String,
        ) {
            properties.setProperty(key, value)
        }

        /**
         * Get the stack trace of a throwable as a string.
         * @param throwable The throwable to get the stack trace of.
         * @return The stack trace as a string.
         */
        @JvmStatic
        fun getStackTrace(throwable: Throwable): String = ExceptionUtils.getStackTrace(throwable)
    }
}
