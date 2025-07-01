package com.bodosql.calcite.adapter.snowflake

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
        fun snowflakeYesNoToBoolean(s: String): Boolean =
            when (val loweredStr = s.uppercase()) {
                "Y" -> true
                "N" -> false
                else -> throw RuntimeException(
                    "Internal Processing Error: Snowflake column expected to hold Y/N but a different value '$loweredStr' was encountered",
                )
            }

        /**
         * Helper function to parse the arguments value of show functions
         * into a form that can be passed to describe function.
         * https://docs.snowflake.com/en/sql-reference/sql/show-functions
         *
         * The signature has the form
         * FUNC_NAME(TYPE_1, ..., TYPE_N, DEFAULT TYPE_N+1, DEFAULT TYPE_N+M) RETURN RETURN_TYPE
         *
         * While we cannot support the default values in Snowflake, we can handle optional arguments if
         * they are provided (and produce a higher quality error message about which arguments are missing).
         *
         * Note that a builtin function can have multiple signatures in general, but we only care about
         * UDFs.
         *
         * @param arguments The output of the arguments column for show functions.
         * @param functionName The name of the function in a case-sensitive format.
         * @return A pair of values containing the signature to pass to describe function
         * and the number of optional arguments (because optional arguments must always come
         * at the end).
         */
        @JvmStatic
        fun parseSnowflakeShowFunctionsArguments(
            arguments: String,
            functionName: String,
        ): Pair<String, Int> {
            // Remove the return
            val callParts = arguments.split(" RETURN ")
            if (callParts.size != 2) {
                throw java.lang.RuntimeException(
                    "UDF formatting error in parseSnowflakeShowFunctionsArguments: " +
                        "expected to split into function definition and return type",
                )
            }
            val call = callParts[0]

            val openParenIndex = call.indexOf('(')
            val closeParenIndex = call.lastIndexOf(')')

            if (openParenIndex == -1 || closeParenIndex == -1 || closeParenIndex <= openParenIndex) {
                throw java.lang.RuntimeException(
                    "UDF formatting error in parseSnowflakeShowFunctionsArguments: " +
                        "expected signature to look like: 'FUNC(ARGS)', got: '$call'",
                )
            }
            val argString = call.substring(openParenIndex + 1, closeParenIndex)
            val (argList, numOptional) =
                if (argString.isEmpty()) {
                    Pair(listOf(), 0)
                } else {
                    var numOptional = 0
                    val args =
                        argString.split(",").map {
                            val trimmedArg = it.trim()
                            val spacedArg = trimmedArg.split(" ")
                            if (spacedArg.size == 1) {
                                trimmedArg
                            } else {
                                if (spacedArg.size != 2 || spacedArg[0].trim() != "DEFAULT") {
                                    throw java.lang.RuntimeException(
                                        "UDF formatting error in parseSnowflakeShowFunctionsArguments: " +
                                            "unexpected special characters",
                                    )
                                }
                                numOptional += 1
                                spacedArg[1]
                            }
                        }
                    Pair(args, numOptional)
                }
            val newSignature = "$functionName(${argList.joinToString()})"
            return Pair(newSignature, numOptional)
        }
    }
}
