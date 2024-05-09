package com.bodosql.calcite.adapter.bodo

import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.type.SqlTypeName

interface BodoCostEstimator {
    companion object {
        /**
         * This is copied from RelMdSize in Calcite with some suitable defaults for
         * value sizes based on the sql type.
         */
        @JvmStatic
        fun averageTypeValueSize(type: RelDataType): Double =
            when (type.sqlTypeName) {
                SqlTypeName.BOOLEAN -> 0.125
                SqlTypeName.TINYINT -> 1.0
                SqlTypeName.SMALLINT -> 2.0
                SqlTypeName.INTEGER, SqlTypeName.FLOAT, SqlTypeName.DATE, SqlTypeName.TIME, SqlTypeName.TIME_WITH_LOCAL_TIME_ZONE -> 4.0
                SqlTypeName.BIGINT, SqlTypeName.DOUBLE, SqlTypeName.DECIMAL, SqlTypeName.REAL, SqlTypeName.TIMESTAMP,
                SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE, SqlTypeName.INTERVAL_DAY, SqlTypeName.INTERVAL_DAY_HOUR,
                SqlTypeName.INTERVAL_DAY_MINUTE, SqlTypeName.INTERVAL_DAY_SECOND, SqlTypeName.INTERVAL_HOUR,
                SqlTypeName.INTERVAL_HOUR_MINUTE, SqlTypeName.INTERVAL_HOUR_SECOND, SqlTypeName.INTERVAL_MINUTE,
                SqlTypeName.INTERVAL_MINUTE_SECOND, SqlTypeName.INTERVAL_SECOND, SqlTypeName.INTERVAL_YEAR,
                SqlTypeName.INTERVAL_YEAR_MONTH, SqlTypeName.INTERVAL_MONTH,
                -> 8.0
                SqlTypeName.BINARY ->
                    if (type.precision <= 0) {
                        DEFAULT_VARBINARY_PRECISION
                    } else {
                        type.precision.toDouble()
                    }

                SqlTypeName.VARBINARY ->
                    if (type.precision <= 0) {
                        DEFAULT_VARBINARY_PRECISION
                    } else {
                        type.precision.toDouble()
                    }

                SqlTypeName.CHAR ->
                    if (type.precision <= 0) {
                        DEFAULT_VARCHAR_PRECISION
                    } else {
                        type.precision.toDouble()
                    }

                SqlTypeName.VARCHAR -> // Even in large (say VARCHAR(2000)) columns most strings are small.
                    // By default, Snowflake gives a very large string value, so we map this case to 16 bytes
                    // rather than just assigning to the max.
                    (
                        if (type.precision <= 0 || type.precision > 1000000) {
                            DEFAULT_VARCHAR_PRECISION
                        } else {
                            type.precision.toDouble()
                        }
                    ).coerceAtMost(100.0)

                // Note: this is the Variant type.
                // TODO: Refactor this to look at actual types instead of type names.
                SqlTypeName.OTHER -> VARIANT_SIZE_BYTES

                // The average row size for an ARRAY is the average row size of the inner elements
                // times the average number of array entries per row.
                SqlTypeName.ARRAY -> {
                    AVG_ARRAY_ENTRIES_PER_ROW * averageTypeValueSize(type.componentType!!)
                }

                // The average row size for a MAP is the average row size its key type
                // plus its value type, all times the average number of array entries per row.
                SqlTypeName.MAP -> {
                    AVG_JSON_ENTRIES_PER_ROW * (averageTypeValueSize(type.keyType!!) + averageTypeValueSize(type.valueType!!))
                }

                SqlTypeName.ROW -> {
                    var average = 0.0
                    for (field in type.fieldList) {
                        average += averageTypeValueSize(field.type)
                    }
                    average
                }
                // Set a reasonable default.
                else -> 8.0
            }

        /**
         * Default precision for a VARCHAR that has no precision set
         * for the purposes of estimating sizes.
         *
         * Number is arbitrary. This seems like a suitable number for
         * something that will likely not be too long but also is still
         * significant for memory use.
         */
        private const val DEFAULT_VARCHAR_PRECISION: Double = 16.0

        /**
         * Default precision for a VARBINARY that has no precision set
         * for the purposes of estimating sizes.
         *
         * Number is arbitrary. This seems like a suitable number for
         * something that will likely not be too long but also is still
         * significant for memory use.
         */
        private const val DEFAULT_VARBINARY_PRECISION: Double = 16.0

        // TODO: Tune this number based on actual data.
        private val VARIANT_SIZE_BYTES: Double = 64.0

        /**
         * Magic number used to guess how many inner elements
         * each row in a column with ARRAY type contains on average.
         */
        public val AVG_ARRAY_ENTRIES_PER_ROW: Double = 16.0

        /**
         * Magic number used to guess how many key-value pairs
         * each row in a column with JSON type contains on average.
         */
        public val AVG_JSON_ENTRIES_PER_ROW: Double = 16.0
    }
}
