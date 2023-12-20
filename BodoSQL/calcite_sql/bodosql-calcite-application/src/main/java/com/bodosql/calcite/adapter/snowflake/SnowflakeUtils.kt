package com.bodosql.calcite.adapter.snowflake

import java.lang.RuntimeException

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
    }
}
