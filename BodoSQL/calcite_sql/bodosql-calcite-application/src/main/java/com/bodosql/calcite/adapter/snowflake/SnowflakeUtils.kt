package com.bodosql.calcite.adapter.snowflake

import java.util.*

/**
 * Utility functions for parsing information obtained from Snowflake. This is intended
 * to be shared across various modules and testing files that all interact with Snowflake.
 */
class SnowflakeUtils {
    companion object {
        /**
         * Helper function to convert a snowflake column that returns
         * either Y or N as a boolean representation of True/False and
         * converts it to a boolean, raising an exception if the input
         * is incorrectly formatted.
         */
        @JvmStatic
        fun snowflakeYesNoToBoolean(s: String): Boolean {
            return when (val loweredStr = s.uppercase()) {
                "Y" -> true
                "N" -> false
                else -> throw RuntimeException("Internal Processing Error: Snowflake column expected to hold Y/N but a different value '$loweredStr' was encountered")
            }
        }

        /**
         * Helper function to parse the arguments value of show functions
         * into a form that can be passed to describe function.
         * https://docs.snowflake.com/en/sql-reference/sql/show-functions
         *
         * The signature has the form
         * FUNC_NAME(TYPE_1, ..., TYPE_N, [OPTIONAL_TYPE_1, ..., OPTIONAL_TYPE_N]) RETURN RETURN_TYPE
         *
         * While we cannot support the default values in Snowflake, we can handle optional arguments if
         * they are provided (and produce a higher quality error message about which arguments are missing).
         *
         * Note that a builtin function can have multiple signatures in general, but we only care about
         * UDFs.
         *
         * @param arguments The output of the arguments column for show functions.
         * @return A pair of values containing the signature to pass to describe function
         * and the number of optional arguments (because optional arguments must always come
         * at the end).
         */
        @JvmStatic
        fun parseSnowflakeShowFunctionsArguments(arguments: String): Pair<String, Int> {
            // Remove the return
            val callParts = arguments.split("RETURN")
            val call = callParts[0]
            // Remove any optional arguments
            val argParts = call.split("[")
            val validArgs = argParts[0].trim()
            // If there is no "[" found we didn't have optional args.
            return if (argParts.size == 1) {
                Pair(validArgs, 0)
            } else {
                // Remove the remaining ] and count the number of optional arguments.
                val optionalPart = argParts[1].split("]")[0].trim()
                // Number of optional elements is number of commas + 1.
                val numOptional: Int = optionalPart.count { it == ',' } + 1
                // Recombine the signatures. The ) and space have been removed.
                val newSignature = "$validArgs $optionalPart)"
                Pair(newSignature, numOptional)
            }
        }
    }
}
