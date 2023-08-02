package com.bodosql.calcite.adapter.pandas

import org.apache.calcite.rel.metadata.RelMdSize
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.type.SqlTypeName

interface PandasCostEstimator {
    /**
     * This is copied from RelMdSize in Calcite with some suitable defaults for
     * value sizes based on the sql type.
     */
    fun averageTypeValueSize(type: RelDataType): Double? =
        when (type.sqlTypeName) {
            SqlTypeName.BOOLEAN, SqlTypeName.TINYINT -> 1.0
            SqlTypeName.SMALLINT -> 2.0
            SqlTypeName.INTEGER, SqlTypeName.REAL, SqlTypeName.DECIMAL, SqlTypeName.DATE, SqlTypeName.TIME, SqlTypeName.TIME_WITH_LOCAL_TIME_ZONE, SqlTypeName.INTERVAL_YEAR, SqlTypeName.INTERVAL_YEAR_MONTH, SqlTypeName.INTERVAL_MONTH -> 4.0
            SqlTypeName.BIGINT, SqlTypeName.DOUBLE, SqlTypeName.FLOAT, SqlTypeName.TIMESTAMP, SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE, SqlTypeName.INTERVAL_DAY, SqlTypeName.INTERVAL_DAY_HOUR, SqlTypeName.INTERVAL_DAY_MINUTE, SqlTypeName.INTERVAL_DAY_SECOND, SqlTypeName.INTERVAL_HOUR, SqlTypeName.INTERVAL_HOUR_MINUTE, SqlTypeName.INTERVAL_HOUR_SECOND, SqlTypeName.INTERVAL_MINUTE, SqlTypeName.INTERVAL_MINUTE_SECOND, SqlTypeName.INTERVAL_SECOND -> 8.0
            SqlTypeName.BINARY -> type.precision.coerceAtLeast(1).toDouble()
            SqlTypeName.VARBINARY -> type.precision.coerceAtLeast(1).toDouble().coerceAtMost(100.0)
            SqlTypeName.CHAR -> type.precision.coerceAtLeast(1).toDouble() * RelMdSize.BYTES_PER_CHARACTER
            SqlTypeName.VARCHAR -> // Even in large (say VARCHAR(2000)) columns most strings are small
                (type.precision.coerceAtLeast(1).toDouble() * RelMdSize.BYTES_PER_CHARACTER).coerceAtMost(100.0)

            SqlTypeName.ROW -> {
                var average = 0.0
                for (field in type.fieldList) {
                    val size = averageTypeValueSize(field.type)
                    if (size != null) {
                        average += size
                    }
                }
                average
            }

            else -> null
        }
}
